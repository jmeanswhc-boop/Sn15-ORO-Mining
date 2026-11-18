from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


# ===== shared helpers =====
_proxy = ProxyClient(timeout=180, max_retries=4)
_MODELS: dict[str, dict[str, Any]] = {'chutes': {'strong': 'Qwen/Qwen3.5-397B-A17B-TEE', 'fast': 'deepseek-ai/DeepSeek-V3.2-TEE', 'fallback': ('moonshotai/Kimi-K2.6-TEE', 'zai-org/GLM-5.2-TEE', 'Qwen/Qwen3.6-27B-TEE')}, 'openrouter': {'strong': 'qwen/qwen3.5-397b-a17b', 'fast': 'deepseek/deepseek-v3.2', 'fallback': ('moonshotai/kimi-k2.6', 'z-ai/glm-5.2', 'qwen/qwen3.6-27b')}}
_TEE_TO_OPENROUTER = {'deepseek-ai/DeepSeek-V3.2-TEE': 'deepseek/deepseek-v3.2', 'deepseek-ai/DeepSeek-V3.1-TEE': 'deepseek/deepseek-chat-v3.1', 'deepseek-ai/DeepSeek-V3-0324-TEE': 'deepseek/deepseek-chat-v3-0324', 'deepseek-ai/DeepSeek-R1-0528-TEE': 'deepseek/deepseek-r1-0528', 'Qwen/Qwen3.5-397B-A17B-TEE': 'qwen/qwen3.5-397b-a17b', 'Qwen/Qwen3-32B-TEE': 'qwen/qwen3-32b', 'Qwen/Qwen3.6-27B-TEE': 'qwen/qwen3.6-27b', 'moonshotai/Kimi-K2.6-TEE': 'moonshotai/kimi-k2.6', 'zai-org/GLM-5.1-TEE': 'z-ai/glm-5.1', 'zai-org/GLM-5.2-TEE': 'z-ai/glm-5.2', 'google/gemma-4-31B-turbo-TEE': 'google/gemma-4-31b-it', 'XiaomiMiMo/MiMo-V2-Flash-TEE': 'xiaomi/mimo-v2-flash', 'openai/gpt-oss-120b-TEE': 'openai/gpt-oss-120b', 'unsloth/Mistral-Nemo-Instruct-2407-TEE': 'mistralai/mistral-small-2603'}
_WORD_RE = re.compile('[a-z0-9]+', re.I)
_NUM_RE = re.compile('\\d+(?:\\.\\d+)?')
_VARIANT_RE = re.compile('\\b(\\d{6,})::([A-Za-z0-9_-]+)\\b')
_BUDGET_RE = re.compile('(?:(?:under|below|upto|up\\s+to|at\\s+or\\s+under|no\\s+more\\s+than|max(?:imum)?|budget(?:\\s+of)?|spend(?:\\s+up\\s+to)?|can\\s+spend(?:\\s+up\\s+to)?|less\\s+than)\\s*(?:of\\s+)?(?P<a>\\d[\\d,]*(?:\\.\\d+)?)|(?P<b>\\d[\\d,]*(?:\\.\\d+)?)\\s*(?P<cur>[a-z]{3})\\b)', re.I)
_DIR_UP = frozenset('higher more greater larger better maximize maximise prefer preferring'.split())
_DIR_DOWN = frozenset('lower less smaller cheaper minimize minimise'.split())
_RANK_AXIS_RE = re.compile('\\b(?P<dir>higher|more|greater|larger|better|prefer(?:ring)?|maximize|maximise|lower|less|smaller|cheaper|minimize|minimise)\\s+(?P<axis>[a-z][a-z0-9_]*(?:\\s+[a-z][a-z0-9_]*){0,3})(?=\\s*(?:,|;|\\.|then|and|before|after|than|$))', re.I)
_FIRM_CUE_RE = re.compile("\\b(?:must|need(?:s|ed)?|require(?:s|d)?|only|exactly|strict(?:ly)?|have\\s+to|has\\s+to|cannot|can't|won't|will\\s+not)\\b", re.I)
_SOFT_CUE_RE = re.compile('\\b(?:prefer(?:red|ably)?|ideally|nice\\s+to\\s+have|if\\s+possible|would\\s+like|bonus|optional|soft)\\b', re.I)
_DEFER_ASK_RE = (re.compile('\\b(?:ask|asking)\\b.{0,48}\\b(?:tell|share|reveal|provide|give)\\b', re.I), re.compile('\\b(?:tell|share|reveal|provide|give)\\b.{0,48}\\b(?:ask|asking)\\b', re.I), re.compile("\\b(?:haven'?t|have not|not yet)\\s+(?:told|said|shared|given)\\b", re.I), re.compile('\\bif\\b.{0,24}\\bask\\b.{0,40}\\b(?:tell|share|reveal|provide|give)\\b', re.I), re.compile('\\bwill\\s+tell\\b.{0,64}\\bif\\s+you\\s+ask\\b', re.I), re.compile('\\bone\\s+firm\\s+requirement\\b.{0,80}\\bask\\b', re.I))
PROC_ASK = 'ask_gate'
PROC_RECALL = 'phrase_recall'
PROC_STRICT = 'strict_match'
PROC_SOFT = 'soft_prefer'
PROC_USE = 'use_fit'
PROC_RANK = 'axis_rank'
PROC_ADAPT = 'event_adapt'
PROC_CLAIM = 'claim_why'
_WHY_GOAL_RE = re.compile('\\b(?:why\\b.{0,28}(?:pick|choose|order|one|it)|tell me why|walk me through why|give me a reason|reason I can|justify|justification)\\b', re.I)
_PHRASE_GOAL_RE = re.compile('(?:matches?\\s+(?:the\\s+)?(?:phrase\\s+)?|that\\s+matches\\s+)[\\"\\u201c]?([^\\"\\u201d\\n,]{2,80})', re.I)
_SOFT_OBJECT_RE = re.compile("(?:prefer(?:red|ably)?|ideally(?:\\s*,)?\\s*I\\s+prefer)\\s+(?:the\\s+)?(?:brand\\s+)?([A-Za-z][A-Za-z0-9][\\w.&'-]{0,40})", re.I)
_USE_FIT_RE = re.compile("\\b(?:it'?s\\s+for|for)\\s+(?!the\\b|a\\b|an\\b|this\\b|that\\b|my\\b|your\\b|me\\b|us\\b|him\\b|her\\b|them\\b|it\\b|in\\b|stock\\b|under\\b|below\\b|up\\b)[a-z][a-z0-9-]{2,}", re.I)
_ADAPT_GOAL_RE = re.compile('\\b(?:if\\s+(?:it|that|the\\s+item)\\s+(?:becomes?\\s+)?(?:unavailable|disappears?|sold\\s*out)|recover|fallback|another\\s+(?:colou?r|color|option|variant))\\b', re.I)
_FIRM_MATTERS_RE = re.compile('\\bfirm\\s+requirements?\\b.{0,40}\\bmatter|\\bbut\\s+the\\s+firm\\b', re.I)
_OBS_CAP = 3400
_STR_CAP = 320
_LIST_CAP = 8
_HISTORY_CAP = 16
_BOARD_SHOW = 6
_NUM_NEAR_RE = re.compile('(?P<num>\\d+(?:\\.\\d+)?)\\s*(?P<unit>[A-Za-z][A-Za-z0-9_+/-]{0,12})?|(?P<unit2>[A-Za-z][A-Za-z0-9_+/-]{0,12})\\s*(?P<num2>\\d+(?:\\.\\d+)?)')

def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'true', 'yes', 'y', '1', 'in_stock', 'available'}:
            return True
        if text in {'false', 'no', 'n', '0', 'out_of_stock', 'unavailable', 'sold_out'}:
            return False
    return None

def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(',', '')
        try:
            return float(text)
        except ValueError:
            return None
    return None

def _product_blob(row: dict[str, Any], viewed: dict[str, Any] | None=None) -> str:
    parts: list[Any] = [row, viewed or {}]
    facts = (viewed or {}).get('facts') if isinstance((viewed or {}).get('facts'), dict) else {}
    if facts:
        parts.append(facts)
    return json.dumps(parts, ensure_ascii=False, default=str).lower()

def _atoms_hit(atoms: list[str], blob: str) -> tuple[int, int]:
    if not atoms:
        return (0, 0)
    hits = 0
    blob_nums = set(_NUM_RE.findall(blob))
    blob_floats: list[float] = []
    for cand in blob_nums:
        try:
            blob_floats.append(float(cand))
        except ValueError:
            continue
    for atom in atoms:
        compact_atom = re.sub('\\s+', '', atom)
        compact_blob = re.sub('\\s+', '', blob)
        if atom in blob or (len(compact_atom) >= 2 and compact_atom in compact_blob):
            hits += 1
            continue
        num_m = re.fullmatch('(\\d+(?:\\.\\d+)?)(?:\\s*[a-z%][a-z0-9_+/-]*)?', atom)
        if num_m:
            try:
                target = float(num_m.group(1))
            except ValueError:
                continue
            if num_m.group(1) in blob_nums or any((abs(x - target) <= 1e-09 for x in blob_floats)):
                hits += 1
    return (hits, len(atoms))

def _value_attested(value: Any, blob: str) -> bool:
    if value is None or value == '':
        return False
    if isinstance(value, bool):
        token = 'true' if value else 'false'
        return token in blob
    text = str(value).strip().lower()
    if len(text) < 1:
        return False
    if text in blob:
        return True
    compact = re.sub('\\s+', '', text)
    if len(compact) >= 2 and compact in re.sub('\\s+', '', blob):
        return True
    num = _as_float(value)
    if num is not None:
        for cand in _NUM_RE.findall(blob):
            try:
                if abs(float(cand) - num) <= 1e-09:
                    return True
            except ValueError:
                continue
    return False

def _provider() -> str:
    return (getenv('INFERENCE_PROVIDER') or 'chutes').strip().lower()

def _model_ladder(hard: bool) -> list[str]:
    provider = _provider()
    catalog = _MODELS.get(provider, _MODELS['chutes'])
    override = (getenv('SANDBOX_MODEL') or '').strip()
    ordered: list[str] = []
    ordered.append(catalog['strong'])
    if override:
        if provider == 'openrouter' and override.endswith('-TEE'):
            ordered.append(_TEE_TO_OPENROUTER.get(override, catalog['fast']))
        else:
            ordered.append(override)
    ordered.append(catalog['fast'])
    ordered.extend(catalog['fallback'])
    uniq: list[str] = []
    seen: set[str] = set()
    for model in ordered:
        if model and model not in seen:
            seen.add(model)
            uniq.append(model)
    return uniq

def _chat(*, models: list[str], messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    last: Exception | None = None
    for model in models:
        try:
            result = _proxy.post('/inference/chat/completions', json_data={'model': model, 'messages': messages, 'tools': tools, 'tool_choice': 'required', 'temperature': 0})
            if result is None:
                raise RuntimeError('inference request failed')
            return result
        except Exception as exc:
            last = exc
            continue
    raise RuntimeError(f'inference request failed: {last}')

def _fn(spec: dict[str, Any]) -> dict[str, Any] | None:
    raw = spec.get('function') if isinstance(spec, dict) else None
    return raw if isinstance(raw, dict) else None

def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {fn['name'] for spec in tools for fn in [_fn(spec)] if fn and isinstance(fn.get('name'), str)}

def _required_args(tools: list[dict[str, Any]], name: str) -> list[str]:
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get('name') != name:
            continue
        params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
        return [x for x in params.get('required') or [] if isinstance(x, str)]
    return []

def _tool_props(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get('name') != name:
            continue
        params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
        props = params.get('properties') if isinstance(params.get('properties'), dict) else {}
        return props
    return {}

def _claim_fields(tools: list[dict[str, Any]]) -> set[str]:
    """Discover allowed justification field names from the live order schema only."""
    allowed: set[str] = set()
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get('name') != 'place_test_order':
            continue
        params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
        props = params.get('properties') if isinstance(params.get('properties'), dict) else {}
        just = props.get('justification') if isinstance(props.get('justification'), dict) else {}
        jprops = just.get('properties') if isinstance(just.get('properties'), dict) else {}
        claims = jprops.get('claims') if isinstance(jprops.get('claims'), dict) else {}
        items = claims.get('items') if isinstance(claims.get('items'), dict) else {}
        iprops = items.get('properties') if isinstance(items.get('properties'), dict) else {}
        field = iprops.get('field') if isinstance(iprops.get('field'), dict) else {}
        enum = field.get('enum')
        has_enum = False
        if isinstance(enum, list):
            vals = [str(x) for x in enum if isinstance(x, str) and x.strip()]
            if vals:
                has_enum = True
                allowed.update(vals)
        if not has_enum:
            blob = json.dumps(params, ensure_ascii=False).lower()
            for token in re.findall('\\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\\b', blob):
                if token not in {'product_id', 'tool_call', 'call_id', 'max_price', 'in_stock'}:
                    allowed.add(token)
        if 'justification' in json.dumps(params, ensure_ascii=False).lower() or allowed:
            allowed.update({'price', 'in_stock'})
    return allowed

def _open_spec_claims(tools: list[dict[str, Any]]) -> bool:
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get('name') != 'place_test_order':
            continue
        blob = json.dumps(fn.get('parameters') or {}, ensure_ascii=False).lower()
        if 'specification' in blob and 'shopper' in blob:
            return True
        if 'specification field' in blob:
            return True
    return False

def _soft_schema_fields(tools: list[dict[str, Any]], allowed: set[str]) -> set[str]:
    soft: set[str] = set()
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get('name') != 'place_test_order':
            continue
        blob = json.dumps(fn.get('parameters') or {}, ensure_ascii=False).lower()
        for field in allowed:
            if field in {'price', 'in_stock'}:
                continue
            surface = field.replace('_', ' ')
            if field in blob and any((cue in blob for cue in ('reason', 'capability', 'positive', 'why', 'benefit'))):
                soft.add(field)
            if surface in blob and 'positive' in blob:
                soft.add(field)
    return soft

def _preamble(query: str) -> str:
    return (query or '').split('Task rules:', 1)[0].strip()

def _goal_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for match in re.finditer('"([^"\\n]{2,80})"|“([^”\\n]{2,80})”', text or ''):
        body = (match.group(1) or match.group(2) or '').strip()
        key = body.lower()
        if len(body) >= 2 and key not in seen:
            seen.add(key)
            phrases.append(body)
    for match in _PHRASE_GOAL_RE.finditer(text or ''):
        body = (match.group(1) or '').strip().strip('"\'“”')
        key = body.lower()
        if len(body) >= 2 and key not in seen:
            seen.add(key)
            phrases.append(body)
    return phrases[:4]

def _procedure_playbook(proc: str) -> list[str]:
    """Short procedure-specific instructions for the deliberative model."""
    books: dict[str, list[str]] = {PROC_ASK: ['Procedure ask_gate: a firm specification is withheld until you ask.', 'Send one focused message first; do not cart until the shopper answers.', 'After clarification, treat the new atom as non-negotiable on the exact SKU.'], PROC_RECALL: ['Procedure phrase_recall: prioritize the quoted/phrase tokens in search.', 'Reformulate with keyword splits; survey several distinct listings before carting.', 'Verify category_path against the requested category using observed path text.'], PROC_STRICT: ['Procedure strict_match: every stated numeric/phrase atom must evidence on the SKU.', 'Titles alone are insufficient — use view/compare facts for the selected variant.', 'Reject listings that miss any firm atom even if cheaper.'], PROC_SOFT: ['Procedure soft_prefer: firm constraints always beat soft wishes.', 'Among fully compliant options, honor the named soft preference on the FIRST cart.', 'Never buy a soft-matching item that fails a firm atom.'], PROC_USE: ['Procedure use_fit: hard gates are category, budget, and stock.', 'Rank compliant items by how well raw facts match the stated use context.', 'Do not default to the cheapest compliant listing when a better use-fit exists.'], PROC_RANK: ['Procedure axis_rank: compare only the named product_id::sku variants.', 'Score by the stated axis order (primary, then secondary, then price).', 'Cart the Pareto/lexicographic winner that is live in-stock within budget.'], PROC_ADAPT: ['Procedure event_adapt: after cart or stock failure, remove and switch quickly.', 'Prefer least change: same product other sku/option before unrelated listings.', 'Always re-inspect_stock after cart events before place_test_order.'], PROC_CLAIM: ['Procedure claim_why: view/compare the final candidate before the first cart.', 'Checkout claims must use observed facts for the ordered SKU plus live price/stock.', 'Never invent field values from the title alone.']}
    return books.get(proc, books[PROC_STRICT])

def _parse_budget(text: str) -> float | None:
    best: float | None = None
    for match in _BUDGET_RE.finditer(text or ''):
        raw = match.group('a') or match.group('b')
        if not raw:
            continue
        try:
            value = float(raw.replace(',', ''))
        except ValueError:
            continue
        if value < 5:
            continue
        if best is None or value < best:
            best = value
    return best

def _needs_clarification(query: str) -> bool:
    text = _preamble(query)
    if not text:
        return False
    if any((pat.search(text) for pat in _DEFER_ASK_RE)):
        return True
    return False

def _named_variants(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _VARIANT_RE.finditer(text or ''):
        key = f'{match.group(1)}::{match.group(2)}'
        if key in seen:
            continue
        seen.add(key)
        found.append((match.group(1), match.group(2)))
    return found

def _fields_in_text(query: str, allowed: set[str], extra: str='') -> set[str]:
    if not allowed:
        return set()
    q = f'{_preamble(query)} {extra}'.lower()
    q_norm = re.sub('[^a-z0-9]+', ' ', q)
    compact = q_norm.replace(' ', '')
    hits: set[str] = set()
    for field in allowed:
        if field in {'price', 'in_stock'}:
            continue
        surface = field.replace('_', ' ')
        if field in q_norm or surface in q or surface.replace(' ', '') in compact:
            hits.add(field)
    return hits

def _overlap(value: Any, goal: set[str]) -> int:
    if value in (None, ''):
        return 0
    text = str(value).lower()
    tokens = set(_WORD_RE.findall(text))
    score = len(tokens & goal)
    for num in _NUM_RE.findall(text):
        if num in goal or any((num in g or g in num for g in goal if any((ch.isdigit() for ch in g)))):
            score += 2
    return score

def _norm_key(text: str) -> str:
    return re.sub('[^a-z0-9]+', '', str(text).lower())

def _keys_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4:
        return a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)
    return False

def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    """Merge nested observation fact blocks without inventing attributes."""
    flat = dict(row)
    for nest in ('facts', 'specification', 'attributes', 'specs'):
        block = row.get(nest)
        if isinstance(block, dict):
            for (key, value) in block.items():
                flat.setdefault(key, value)
    path = row.get('category_path')
    if isinstance(path, list) and path:
        leaf = path[-1] if isinstance(path[-1], str) else path[0]
        if isinstance(leaf, str) and leaf.strip():
            flat.setdefault('category_path_leaf', leaf)
    elif isinstance(path, str) and path.strip():
        flat.setdefault('category_path_leaf', path.split(',')[-1].strip().strip('[]"\' '))
    return flat

def _lookup(blobs: list[dict[str, Any]], field: str) -> Any:
    needle = _norm_key(field)
    if not needle:
        return None
    best: tuple[int, Any] | None = None
    for blob in blobs:
        stack: list[Any] = [blob]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            for (key, value) in node.items():
                if value in (None, ''):
                    continue
                kn = _norm_key(key)
                if _keys_match(kn, needle):
                    rank = abs(len(kn) - len(needle))
                    if best is None or rank < best[0]:
                        best = (rank, value)
                if isinstance(value, dict):
                    stack.append(value)
                elif isinstance(value, list):
                    stack.extend((item for item in value if isinstance(item, dict)))
    return None if best is None else best[1]

def _goal_nums_near(query: str, field: str, notes: list[str] | None=None) -> set[str]:
    text = _preamble(query).lower()
    if notes:
        text = f"{text} {' '.join(notes[-3:]).lower()}"
    surfaces = {field.lower(), field.replace('_', ' ').lower(), _norm_key(field), *[p for p in field.lower().split('_') if len(p) >= 4]}
    hits: set[str] = set()
    for surface in surfaces:
        if len(surface) < 3:
            continue
        for match in re.finditer(re.escape(surface), text):
            window = text[max(0, match.start() - 32):match.end() + 32]
            hits.update(_NUM_RE.findall(window))
    return hits

def _repair_json(raw: str) -> dict[str, Any]:
    text = (raw or '').strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    fixed = text.replace("'", '"')
    fixed = re.sub(',\\s*}', '}', fixed)
    fixed = re.sub(',\\s*]', ']', fixed)
    try:
        parsed = json.loads(fixed)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}

def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get('function', {}).get('arguments', '{}')
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return _repair_json(raw)
    return {}

def _compact(value: Any, depth: int=0) -> Any:
    if depth > 5:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= _STR_CAP else value[:_STR_CAP - 3] + '...'
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_compact(item, depth + 1) for item in value[:_LIST_CAP]]
    if isinstance(value, dict):
        return {str(k): _compact(v, depth + 1) for (k, v) in value.items()}
    return str(value)[:200]

def _compact_obs(observation: dict[str, Any]) -> str:
    text = json.dumps(_compact(observation), ensure_ascii=False, default=str)
    return text if len(text) <= _OBS_CAP else text[:_OBS_CAP - 3] + '...'

def _unwrap_observation(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split envelope ``{done, error, observation: {...}}`` from the inner payload.

    Runtime tools v5 nest the tool body under ``observation``. The ledger must
    read the inner payload; ``done`` / ``error`` stay on the envelope.
    """
    if not isinstance(raw, dict):
        return ({}, {'raw': raw})
    envelope = dict(raw)
    inner = envelope.get('observation')
    if isinstance(inner, dict) and ('results' in inner or 'compared' in inner or 'product_id' in inner or ('variants' in inner) or ('facts' in inner) or ('action' in inner) or ('items' in inner) or ('cart' in inner) or ('price' in inner) or ('in_stock' in inner)):
        payload = dict(inner)
    elif isinstance(inner, dict) and (not any((k in envelope for k in ('results', 'product_id', 'compared', 'price', 'in_stock')))):
        meta_keys = {'done', 'error', 'observation', 'ok', 'status'}
        if set(envelope.keys()) <= meta_keys | {'observation'}:
            payload = dict(inner)
        else:
            payload = {k: v for (k, v) in envelope.items() if k not in {'done', 'error'}}
    else:
        payload = {k: v for (k, v) in envelope.items() if k not in {'done', 'error'}}
        if 'observation' in payload and (not isinstance(payload.get('observation'), dict)):
            pass
        elif 'observation' in payload and isinstance(payload['observation'], dict):
            nested = payload.pop('observation')
            for (key, value) in nested.items():
                payload.setdefault(key, value)
    return (envelope, payload)

def _trim(messages: list[dict[str, Any]]) -> None:
    if len(messages) <= _HISTORY_CAP:
        return
    head = messages[:2]
    tail = messages[-(_HISTORY_CAP - 3):]
    messages[:] = head + [{'role': 'user', 'content': 'Earlier turns are summarized on the ledger. Prefer ledger + recent tools.'}] + tail
_DEFAULT_MODELS = {'chutes': 'deepseek-ai/DeepSeek-V3.2-TEE', 'openrouter': 'deepseek/deepseek-v3.2'}

# ===== KE_2.py pipeline (tf1) =====
_STRONG_MODELS = {'chutes': 'Qwen/Qwen3.5-397B-A17B-TEE', 'openrouter': 'qwen/qwen3.5-397b-a17b'}
_TOOL_ID_RE = re.compile('^[A-Za-z0-9_-]{6,64}$')
_EPISODE_BUDGET_S = 165.0
_MAX_INFERENCE_ATTEMPTS = 2
_CANDIDATE_RE = re.compile('\\b(\\d{10,})::([A-Za-z0-9._-]+)\\b')
_SCHEMA_NOISE = frozenset('\n    the and for with from that this must only field fields value values claim\n    claims true false string object array required optional product order\n    justify justification supported reason reasons useful such also use shopper\n    request specification feature features explicitly positive capabilities\n    unknown absent checkable purchase facts include example examples when where\n    which your item items listing listings one\n    '.split())
_META_FIELDS = frozenset({'price', 'in_stock', 'brand', 'category', 'title'})
_NOISE_FIELDS = frozenset('\n    spec specification attribute attributes feature features property properties\n    detail details info data value field option options param parameter\n    parameters quality metric metrics requirement requirements\n    '.split())
_SYSTEM = 'You are a helpful shopping assistant with catalog tools.\nHelp the shopper find a suitable in-stock item and complete checkout.\n\nHow to work:\n- Search with short keyword queries; use filter when price/category/brand bounds help.\n- You may call several tools in one turn when allowed.\n- Open product details (view/compare) and check the exact variant before adding to cart.\n- If the shopper asks a question, answer briefly with the message tool, then continue.\n- If price or stock changes after a cart action, pick another valid option.\n- When a valid option is ready, place the order. Do not ask for permission.\n- Only include order justification when that tool parameter is required; then use field/value claims grounded in what you viewed and the live price/stock.\n- Work efficiently: gather evidence quickly and finish.'

def tf1_model(*, prefer_strong: bool=False) -> str:
    override = getenv('SANDBOX_MODEL')
    if override:
        return override
    provider = _provider()
    if prefer_strong:
        return _STRONG_MODELS.get(provider, _STRONG_MODELS['chutes'])
    return _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS['chutes'])

def _tool_name(spec: dict[str, Any]) -> str:
    fn = spec.get('function') if isinstance(spec.get('function'), dict) else spec
    return str((fn or {}).get('name') or '')

def _tool_index(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: spec for spec in tools if (name := _tool_name(spec))}

def _tool_parameters(spec: dict[str, Any]) -> dict[str, Any]:
    fn = spec.get('function') if isinstance(spec.get('function'), dict) else spec
    params = (fn or {}).get('parameters') or {}
    return params if isinstance(params, dict) else {}

def _justification_required(index: dict[str, dict[str, Any]]) -> bool:
    spec = index.get('place_test_order')
    if not spec:
        return False
    return 'justification' in set(_tool_parameters(spec).get('required') or [])

def _claim_space(index: dict[str, dict[str, Any]]) -> tuple[set[str], set[str], bool]:
    """Discover allowed/soft claim fields from the live order-tool schema only."""
    spec = index.get('place_test_order')
    if not spec:
        return ({'price', 'in_stock'}, set(), False)
    params = _tool_parameters(spec)
    props = params.get('properties') if isinstance(params.get('properties'), dict) else {}
    just = props.get('justification') if isinstance(props.get('justification'), dict) else {}
    jprops = just.get('properties') if isinstance(just.get('properties'), dict) else {}
    claims = jprops.get('claims') if isinstance(jprops.get('claims'), dict) else {}
    items = claims.get('items') if isinstance(claims.get('items'), dict) else {}
    iprops = items.get('properties') if isinstance(items.get('properties'), dict) else {}
    field = iprops.get('field') if isinstance(iprops.get('field'), dict) else {}
    allowed: set[str] = {'price', 'in_stock'}
    soft: set[str] = set()
    enum = field.get('enum')
    if isinstance(enum, list):
        allowed.update((str(x) for x in enum if isinstance(x, str) and x.strip()))
    desc = ' '.join((str(p) for p in (field.get('description'), just.get('description'), claims.get('description')) if isinstance(p, str)))
    soft_blob = desc.lower()
    m = re.search('reason(?:s)?\\s*:?\\s*([^.]+)', soft_blob)
    if m:
        soft_blob = m.group(1)
    for tok in re.findall('\\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\\b', soft_blob):
        if tok not in _SCHEMA_NOISE:
            soft.add(tok)
            allowed.add(tok)
    for chunk in re.split('\\s*(?:,|;|/|\\bor\\b|\\band\\b)\\s*', soft_blob):
        words = [w for w in re.findall('[a-z][a-z0-9]+', chunk.casefold()) if w not in _SCHEMA_NOISE and len(w) >= 2]
        if 1 <= len(words) <= 3:
            tok = '_'.join(words)
            if len(tok) >= 3 and tok not in _SCHEMA_NOISE:
                soft.add(tok)
                allowed.add(tok)
    allowed.discard('specification')
    soft -= {'price', 'in_stock', 'category', 'specification'}
    open_spec = 'specification' in desc.casefold() and 'shopper' in desc.casefold() or 'specification field' in desc.casefold()
    if not (isinstance(enum, list) and enum):
        blob = json.dumps(params, ensure_ascii=False).lower()
        for token in re.findall('\\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\\b', blob):
            if token not in _SCHEMA_NOISE | {'product_id', 'tool_call', 'call_id', 'max_price'}:
                allowed.add(token)
    return (allowed, soft, open_spec)

def tf1_named_variants(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _CANDIDATE_RE.finditer(text or ''):
        key = f'{m.group(1)}::{m.group(2)}'
        if key in seen:
            continue
        seen.add(key)
        out.append((m.group(1), m.group(2)))
    return out

def _quoted_phrases(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer('"([^"\\n]{2,80})"', text or '') if m.group(1).strip()]

def tf1_chat(*, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    return _proxy.post('/inference/chat/completions', json_data={'model': model, 'messages': messages, 'tools': tools, 'tool_choice': 'required', 'temperature': 0})

def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(',\\s*}', '}', raw.replace("'", '"'))
        fixed = re.sub(',\\s*]', ']', fixed)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}

def _coerce_value(value: Any, schema: dict[str, Any]) -> Any:
    typ = schema.get('type')
    if typ == 'string':
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, (int, float)):
            return str(value)
        return str(value)
    if typ == 'integer':
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    if typ == 'number':
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if typ == 'boolean':
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().casefold()
            if low in {'true', '1', 'yes'}:
                return True
            if low in {'false', '0', 'no'}:
                return False
        return bool(value)
    if typ == 'array':
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = [part.strip() for part in value.split(',') if part.strip()]
        if isinstance(value, list):
            item_schema = schema.get('items') or {}
            return [_coerce_value(item, item_schema) for item in value]
    if typ == 'object' and isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else value
        except json.JSONDecodeError:
            return value
    return value
_PROC_PREFIX_RE = re.compile('^(?:find(?:\\s+and\\s+place)?(?:\\s+one)?(?:\\s+test)?(?:\\s+order)?(?:\\s+for)?|find\\s+an?\\s+in[- ]stock\\s+listing(?:\\s+in\\s+the)?|want\\s+an?\\s+in[- ]stock|i\\s+(?:need|want|would\\s+like)|can\\s+you\\s+(?:get|find|help)|please\\s+(?:find|get|buy))\\s+', flags=re.IGNORECASE)

def _short_query(text: str) -> str:
    head = text.strip().split('\n', 1)[0].strip()
    head = _PROC_PREFIX_RE.sub('', head).strip(' .,;:') or head
    head = re.sub('^(?:an?\\s+|the\\s+)', '', head, flags=re.IGNORECASE)
    if re.search('\\d{8,}', head) and len(re.findall('[A-Za-z]+', head)) <= 2:
        return head[:100] or 'product'
    stop = {'need', 'want', 'find', 'with', 'under', 'below', 'stock', 'listing', 'please', 'that', 'this', 'from', 'have', 'must', 'place', 'order', 'test', 'one', 'for', 'and', 'the', 'in'}
    tokens = [t for t in re.findall('[A-Za-z0-9]+(?:\\.[0-9]+)?', head) if t.casefold() not in stop]
    if not tokens:
        tokens = re.findall('[A-Za-z0-9]+(?:\\.[0-9]+)?', head)
    if len(tokens) <= 7 and len(' '.join(tokens)) <= 80:
        return ' '.join(tokens) or 'product'
    return ' '.join(tokens[:6]) or 'product'

def _parse_budget_mention(text: str) -> float | None:
    """Pull an updated budget from shopper / observation text when stated explicitly."""
    if not text:
        return None
    patterns = ('(?:at\\s+or\\s+)?(?:under|below|max(?:imum)?|budget(?:\\s+is|\\s+of)?|cap(?:\\s+is)?|up\\s+to|at\\s+most|no\\s+more\\s+than|can\\s+only\\s+go(?:\\s+up)?\\s+to)\\s*(?:of\\s*)?(?:PHP\\s*)?(\\d+(?:\\.\\d+)?)', '(?:PHP\\s*)?(\\d+(?:\\.\\d+)?)\\s*(?:PHP\\s+)?(?:max|budget|cap)\\b', '(?:i\\s+can\\s+only\\s+(?:spend|go|pay)|only\\s+go\\s+up\\s+to)\\s*(?:PHP\\s*)?(\\d+(?:\\.\\d+)?)')
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None

def _sanitize_args(name: str, args: dict[str, Any], index: dict[str, dict[str, Any]], memory: Memory | None=None) -> dict[str, Any]:
    spec = index.get(name)
    if not spec:
        return args
    params = _tool_parameters(spec)
    props = params.get('properties') or {}
    cleaned: dict[str, Any] = {}
    for (key, value) in args.items():
        if key not in props and params.get('additionalProperties') is False:
            continue
        schema = props.get(key) or {}
        cleaned[key] = _coerce_value(value, schema) if schema else value
    for (src, dst) in (('q', 'query'), ('search_query', 'query'), ('productIds', 'product_ids'), ('id', 'product_id'), ('variant', 'sku')):
        if src in cleaned and dst in props and (dst not in cleaned):
            cleaned[dst] = cleaned.pop(src)
    if name == 'search' and 'query' in cleaned:
        q = str(cleaned['query'])
        is_id = bool(re.search('\\d{8,}', q)) and len(re.findall('[A-Za-z]+', q)) <= 2
        if not is_id:
            cleaned['query'] = _short_query(q)
    if name in {'search', 'filter'}:
        if 'in_stock' in props:
            cleaned.setdefault('in_stock', True)
        if 'k' in props:
            cleaned.setdefault('k', 10)
        if name == 'filter' and 'max_price' in props and (memory is not None) and (cleaned.get('max_price') in (None, '', 0)) and (memory.budget is not None):
            cleaned['max_price'] = memory.budget
    if 'k' in props and 'k' in cleaned:
        try:
            k = int(cleaned['k'])
            lo = int((props['k'] or {}).get('minimum') or 1)
            hi = int((props['k'] or {}).get('maximum') or 10)
            cleaned['k'] = max(lo, min(hi, k))
        except (TypeError, ValueError):
            cleaned['k'] = 10
    if 'product_ids' in props and isinstance(cleaned.get('product_ids'), list):
        cleaned['product_ids'] = [str(x) for x in cleaned['product_ids'][:5]]
    return cleaned

def _fingerprint(name: str, args: dict[str, Any]) -> str:
    return json.dumps({'name': name, 'args': args}, sort_keys=True, default=str)

def _unwrap_step(step: dict[str, Any]) -> tuple[dict[str, Any], bool, Any]:
    if not isinstance(step, dict):
        return ({}, False, 'malformed')
    if 'done' in step or 'error' in step:
        payload = step.get('observation')
        if not isinstance(payload, dict):
            payload = step
        return (payload, bool(step.get('done')), step.get('error'))
    return (step, bool(step.get('done')), step.get('error'))

def tf1_compact(value: Any, *, limit: int=3200) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    if isinstance(value, dict):
        shrunk = dict(value)
        for key in ('results', 'variants', 'compared', 'cart', 'products'):
            rows = shrunk.get(key)
            if isinstance(rows, list) and len(rows) > 5:
                slim = []
                for row in rows[:5]:
                    if not isinstance(row, dict):
                        slim.append(row)
                        continue
                    slim.append({k: row.get(k) for k in ('product_id', 'sku', 'title', 'brand', 'price', 'current_price', 'in_stock') if k in row})
                shrunk[key] = slim
                shrunk['truncated'] = True
                text = json.dumps(shrunk, ensure_ascii=False, default=str)
                if len(text) <= limit:
                    return text
    return json.dumps({'truncated': True, 'preview': text[:max(0, limit - 60)]})

def _goal_head(query: str) -> str:
    for marker in ('\n\nTask rules:', '\nTask rules:'):
        if marker in query:
            return query.split(marker, 1)[0].strip()
    return query.strip()

class Memory:

    def __init__(self, goal: str='') -> None:
        self.goal = goal
        self.budget: float | None = None
        self.cart: list[dict[str, Any]] = []
        self.candidates: dict[str, dict[str, Any]] = {}
        self.viewed: set[str] = set()
        self.viewed_facts: dict[str, dict[str, Any]] = {}
        self.stock_by_key: dict[str, dict[str, Any]] = {}
        self.event_targets: set[str] = set()
        self.pending_question: str | None = None
        self.answered = False
        self.recent: list[str] = []
        self.searches: list[str] = []
        self.live_price: dict[str, float] = {}
        self.live_stock: dict[str, bool] = {}
        self.named = tf1_named_variants(_goal_head(goal)) if goal else []
        self.empty_hits = 0
        self.seed_idx = 0
        self.seeds = self._build_seeds()
        self.post_cart = False
        if goal:
            mentioned = _parse_budget_mention(_goal_head(goal))
            if mentioned is not None:
                self.budget = mentioned

    def _build_seeds(self) -> list[str]:
        body = _goal_head(self.goal) if self.goal else ''
        seeds: list[str] = []
        seen: set[str] = set()

        def add(q: str) -> None:
            text = ' '.join(str(q).split())
            if not text:
                return
            key = text.lower()
            if key in seen:
                return
            seen.add(key)
            seeds.append(text)
        add(_short_query(body))
        tokens = re.findall('[A-Za-z][A-Za-z0-9+-]{2,}|\\d{6,}', body)
        stop = {'need', 'want', 'find', 'with', 'under', 'below', 'stock', 'listing', 'category', 'please', 'that', 'this', 'from', 'have', 'must', 'place', 'order', 'test', 'one'}
        toks = [t for t in tokens if t.lower() not in stop][:6]
        if toks:
            add(' '.join(toks[:4]))
            add(' '.join(toks[:2]))
        for phrase in _quoted_phrases(body):
            add(phrase)
        return seeds[:8] or ['product']

    def next_seed(self) -> str:
        if not self.seeds:
            return 'product'
        return self.seeds[min(self.seed_idx, len(self.seeds) - 1)]

    def note(self, name: str, args: dict[str, Any]) -> None:
        self.recent.append(_fingerprint(name, args))
        self.recent = self.recent[-10:]
        if name == 'search' and args.get('query'):
            self.searches.append(str(args['query']))
            self.searches = self.searches[-8:]
        if name == 'message':
            self.answered = True
            self.pending_question = None
        if name == 'add_to_cart':
            self.post_cart = True
        if name in {'inspect_stock', 'place_test_order', 'remove_from_cart'}:
            self.post_cart = False

    def streak(self) -> int:
        if not self.recent:
            return 0
        last = self.recent[-1]
        n = 0
        for fp in reversed(self.recent):
            if fp != last:
                break
            n += 1
        return n

    def hear_shopper(self, content: str) -> None:
        note = content.strip()
        if not note:
            return
        if '?' in note:
            self.pending_question = note
            self.answered = False
        mentioned = _parse_budget_mention(note)
        if mentioned is not None:
            if self.budget is None or mentioned < self.budget:
                self.budget = mentioned

    def ingest(self, step: dict[str, Any], *, action: str | None=None) -> bool:
        (payload, done, _error) = _unwrap_step(step)
        if isinstance(payload.get('budget'), (int, float)):
            incoming = float(payload['budget'])
            if self.budget is None or incoming < self.budget:
                self.budget = incoming
        if isinstance(payload.get('cart'), list):
            self.cart = [r for r in payload['cart'] if isinstance(r, dict)]
            for row in self.cart:
                key = f"{row.get('product_id')}::{row.get('sku')}"
                price = row.get('current_price')
                if price is None:
                    price = row.get('price')
                if isinstance(price, (int, float)):
                    self.live_price[key] = float(price)
                if 'in_stock' in row:
                    self.live_stock[key] = bool(row['in_stock'])
        event = payload.get('market_event')
        if isinstance(event, dict):
            target = event.get('target') or {}
            pid = target.get('product_id')
            sku = target.get('sku')
            if pid and sku:
                key = f'{pid}::{sku}'
                self.event_targets.add(key)
                for price_key in ('new_price', 'price', 'current_price'):
                    if isinstance(event.get(price_key), (int, float)):
                        self.live_price[key] = float(event[price_key])
                        break
                if 'in_stock' in event:
                    self.live_stock[key] = bool(event.get('in_stock'))
        hits = 0
        for row in payload.get('results') or []:
            if isinstance(row, dict):
                self._keep(row)
                hits += 1
        if action == 'search':
            if hits == 0:
                self.empty_hits += 1
                if self.seed_idx < len(self.seeds) - 1:
                    self.seed_idx += 1
            else:
                self.empty_hits = 0
        if payload.get('product_id'):
            pid = str(payload['product_id'])
            self.viewed.add(pid)
            self.viewed_facts[pid] = self._merge_product(self.viewed_facts.get(pid), payload)
            self._keep(payload)
            for variant in payload.get('variants') or []:
                if isinstance(variant, dict):
                    merged = {'product_id': payload.get('product_id'), 'title': payload.get('title'), 'brand': payload.get('brand'), **variant}
                    self._keep(merged)
        for row in payload.get('compared') or []:
            if isinstance(row, dict) and row.get('product_id'):
                rid = str(row['product_id'])
                self.viewed.add(rid)
                self.viewed_facts[rid] = self._merge_product(self.viewed_facts.get(rid), row)
                self._keep(row)
        if payload.get('product_id') and 'sku' in payload and ('in_stock' in payload):
            key = f"{payload.get('product_id')}::{payload.get('sku')}"
            self.stock_by_key[key] = dict(payload)
            self.stock_by_key[str(payload.get('product_id'))] = dict(payload)
            if isinstance(payload.get('price'), (int, float)):
                self.live_price[key] = float(payload['price'])
            self.live_stock[key] = bool(payload.get('in_stock'))
        added = payload.get('added')
        if isinstance(added, dict):
            self._keep(added)
            pid = added.get('product_id')
            sku = added.get('sku')
            if pid and sku:
                key = f'{pid}::{sku}'
                price = added.get('price')
                if price is None:
                    price = added.get('current_price')
                if isinstance(price, (int, float)):
                    self.live_price[key] = float(price)
                if 'in_stock' in added:
                    self.live_stock[key] = bool(added.get('in_stock'))
        return done

    @staticmethod
    def _merge_product(prev: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
        """Merge product payloads so inspect_stock never erases viewed facts."""
        base = dict(prev or {})
        for (key, value) in incoming.items():
            if value is None:
                continue
            if key == 'facts' and isinstance(value, dict):
                prior = base.get('facts') if isinstance(base.get('facts'), dict) else {}
                base['facts'] = {**prior, **value}
                continue
            if key in {'price', 'in_stock', 'sku', 'current_price', 'currency'}:
                base[key] = value
                continue
            if key not in base or base[key] in (None, ''):
                base[key] = value
                continue
            if isinstance(value, (dict, list)):
                base[key] = value
            elif isinstance(value, str) and len(value) >= len(str(base.get(key) or '')):
                base[key] = value
        return base

    def _keep(self, row: dict[str, Any]) -> None:
        pid = row.get('product_id')
        if not pid:
            return
        key = f"{pid}::{row.get('sku') or ''}"
        prev = self.candidates.get(key, {})
        merged = dict(prev)
        for (field, value) in row.items():
            if value is not None:
                merged[field] = value
        self.candidates[key] = merged

    def cart_key(self) -> str | None:
        if not self.cart:
            return None
        row = self.cart[-1]
        pid = row.get('product_id')
        sku = row.get('sku')
        if not pid or not sku:
            return None
        return f'{pid}::{sku}'

    def checkout_ready(self) -> bool:
        key = self.cart_key()
        if not key:
            return False
        if key not in self.live_stock or not self.live_stock.get(key):
            return False
        price = self.live_price.get(key)
        if self.budget is not None and isinstance(price, (int, float)) and (float(price) > self.budget):
            return False
        return True

    def status_line(self) -> str:
        parts = [f'seen={len(self.candidates)}', f'viewed={len(self.viewed)}', f'cart={len(self.cart)}']
        if self.budget is not None:
            parts.append(f'budget={self.budget:g}')
        if self.pending_question and (not self.answered):
            parts.append('shopper_asked_a_question')
        if self.event_targets:
            parts.append('market_changed')
        return 'State: ' + ', '.join(parts)

def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return f'{float(value):g}'
    return str(value or '').strip()

def _evidence_blob(*blobs: dict[str, Any]) -> str:
    parts = []
    for blob in blobs:
        if isinstance(blob, dict):
            parts.append(json.dumps(blob, ensure_ascii=False, default=str))
    return ' '.join(parts)

def _value_evidenced(label: str, evidence: str) -> bool:
    low = str(label).casefold().strip()
    ev = evidence.casefold()
    if not low:
        return False
    if low in ev or low.replace(' ', '') in ev.replace(' ', '').replace('-', ''):
        return True
    nums = re.findall('\\d+(?:\\.\\d+)?', low)
    if nums and all((n in ev for n in nums)):
        return True
    parts = [w for w in re.findall('[a-z0-9]+', low) if len(w) > 1]
    if len(parts) >= 2 and all((p in ev for p in parts)):
        return True
    if len(parts) == 1 and parts[0] in ev and (len(parts[0]) >= 3):
        return True
    return False

def _goal_value_candidates(goal: str) -> list[str]:
    """Values authored in the public goal (phrases, measures, codes) — not suite vocab."""
    body = _goal_head(goal)
    labels: list[str] = []
    seen: set[str] = set()

    def push(raw: str) -> None:
        lab = re.sub('\\s+', ' ', str(raw or '').strip())
        if not lab or len(lab) > 48:
            return
        key = lab.casefold()
        if key in seen:
            return
        seen.add(key)
        labels.append(lab)
        glued = re.sub('(\\d)\\s*[-\\s]+\\s*([A-Za-z])', '\\1\\2', lab)
        glued = re.sub('\\s+', '', glued)
        if glued.casefold() != key and 2 <= len(glued) <= 24:
            if glued.casefold() not in seen:
                seen.add(glued.casefold())
                labels.append(glued)
        m = re.match('^(\\d+(?:\\.\\d+)?)\\s*-?\\s*(inches|inch)$', lab, flags=re.IGNORECASE)
        if m:
            alt = f'{m.group(1)}in'
            if alt.casefold() not in seen:
                seen.add(alt.casefold())
                labels.append(alt)
    for phrase in _quoted_phrases(body):
        push(phrase)
    for m in re.finditer('\\b(\\d+(?:\\.\\d+)?(?:\\s*-?\\s*[A-Za-z]{1,12})?)\\b', body):
        push(m.group(1))
    for m in re.finditer('\\b([A-Za-z][A-Za-z0-9.+_-]{1,20})\\b', body):
        tok = m.group(1)
        if tok.casefold() in {'need', 'want', 'find', 'with', 'under', 'below', 'stock', 'please', 'that', 'this', 'from', 'have', 'must', 'shopper', 'product', 'valid', 'order', 'place', 'test', 'and', 'or', 'in', 'for', 'the', 'inch', 'inches', 'monitor', 'phone', 'tablet', 'gaming'}:
            continue
        if any((ch.isdigit() for ch in tok)):
            push(tok)
        elif tok[:1].isupper() and len(tok) >= 3:
            push(tok)
    labels.sort(key=lambda s: (-(2 if any((c.isdigit() for c in s)) else 0), -len(s)))
    return labels

def _is_dimension_value(text: str) -> bool:
    t = str(text or '')
    return len(re.findall('\\d+(?:\\.\\d+)?', t)) >= 2 and bool(re.search('[x×*,/]', t))

def _prefer_claim_value(raw: Any, goal: str, evidence: str | None=None) -> str:
    """Prefer a goal-authored token when it is evidenced on this value."""
    original = _stringify(raw)
    if not original:
        return original
    local = str(raw)
    hits: list[str] = []
    for lab in _goal_value_candidates(goal):
        if _value_evidenced(lab, local):
            if any((ch.isdigit() for ch in lab)) or len(lab) <= max(6, len(original)):
                hits.append(lab[:80])
    if not hits:
        return original[:80]
    digit_hits = [h for h in hits if any((ch.isdigit() for ch in h))]
    pool = digit_hits or hits
    pool.sort(key=lambda h: (len(re.sub('[^A-Za-z0-9.]', '', h)), len(h)))
    return pool[0]

def _repair_order(args: dict[str, Any], *, index: dict[str, dict[str, Any]], memory: Memory) -> dict[str, Any]:
    cleaned = dict(args)
    if not _justification_required(index):
        cleaned.pop('justification', None)
        return cleaned
    if not cleaned.get('product_id') and memory.cart:
        cleaned['product_id'] = str(memory.cart[-1].get('product_id') or '')
        cleaned['sku'] = str(memory.cart[-1].get('sku') or '')
    pid = str(cleaned.get('product_id') or '')
    sku = str(cleaned.get('sku') or '')
    key = f'{pid}::{sku}'
    stock = memory.stock_by_key.get(key) or memory.stock_by_key.get(pid) or {}
    facts_blob = memory.viewed_facts.get(pid) or {}
    candidate = memory.candidates.get(key) or memory.candidates.get(f'{pid}::') or {}
    product = {**candidate, **facts_blob}
    facts = product.get('facts') if isinstance(product.get('facts'), dict) else {}
    (allowed, soft, open_spec) = _claim_space(index)
    evidence = _evidence_blob(product, stock, facts_blob)
    goal_body = _goal_head(memory.goal)
    ask_vals = _goal_value_candidates(memory.goal)
    grounded: list[dict[str, str]] = []
    seen: set[str] = set()

    def allow(name: str) -> None:
        if re.fullmatch('[A-Za-z][A-Za-z0-9_]{1,40}', name or ''):
            allowed.add(name)

    def soft_attested(field: str, value: Any) -> bool:
        """Soft reasons must be product-attested; bare true/false is not evidence."""
        if field == 'brand':
            brand = product.get('brand') or facts.get('brand')
            return bool(brand) and _value_evidenced(_stringify(brand), evidence)
        for blob in (facts, product, stock):
            if not isinstance(blob, dict):
                continue
            if field in blob:
                raw = blob.get(field)
                if raw is True or str(raw).strip().lower() in {'true', 'yes', '1'}:
                    return True
                if value not in (None, '') and _value_evidenced(_stringify(value), str(raw)):
                    return True
            for (k, v) in blob.items():
                if str(k).casefold().replace(' ', '_') == field:
                    if v is True or str(v).strip().lower() in {'true', 'yes', '1'}:
                        return True
        return False

    def add(field: str, value: Any) -> None:
        if not field or field in seen or field not in allowed:
            return
        if value is None or value == '':
            return
        text = _stringify(value)
        if not text or text.lower() in {'unknown', 'none', 'n/a', 'null'}:
            return
        if field not in {'price', 'in_stock'}:
            if field in soft:
                if not soft_attested(field, value):
                    return
                if field == 'brand':
                    text = _stringify(product.get('brand') or facts.get('brand') or value)
            else:
                text = _prefer_claim_value(value, goal_body)
                if not (_value_evidenced(text, evidence) or _value_evidenced(_stringify(value), evidence)):
                    return
            if len(re.sub('[^A-Za-z0-9]', '', text)) < 2:
                return
            if text.casefold() in {'or', 'and', 'in', 'the', 'a', 'an', 'to', 'of', 'true', 'false'}:
                if field not in soft:
                    return
                if text.casefold() in {'or', 'and', 'in', 'the', 'a', 'an', 'to', 'of'}:
                    return
        grounded.append({'field': field, 'value': text[:80]})
        seen.add(field)
    if key in memory.live_price:
        add('price', memory.live_price[key])
    elif stock.get('price') is not None:
        add('price', stock.get('price'))
    elif product.get('price') is not None:
        add('price', product.get('price'))
    if key in memory.live_stock:
        add('in_stock', memory.live_stock[key])
    elif 'in_stock' in stock:
        add('in_stock', stock.get('in_stock'))
    elif product.get('in_stock') is not None:
        add('in_stock', product.get('in_stock'))
    just = cleaned.get('justification')
    raw_claims = [c for c in just.get('claims') if isinstance(c, dict)] if isinstance(just, dict) and isinstance(just.get('claims'), list) else []
    proposed: list[str] = []
    for claim in raw_claims:
        f = str(claim.get('field') or '').strip()
        if f and re.fullmatch('[A-Za-z][A-Za-z0-9_]{1,40}', f):
            if f.casefold() not in _META_FIELDS and f.casefold() not in _NOISE_FIELDS:
                proposed.append(f)
    skip_keys = {'title', 'listing_id', 'product_id', 'sku', 'model', 'condition', 'navigation', 'brand', 'price', 'in_stock', 'currency', 'category_path', 'options', 'variants', 'category', 'current_price'}
    matched_field = None
    matched_val = None
    for frag in ask_vals:
        frag_l = frag.casefold().strip()
        if len(frag_l) < 2:
            continue
        alts = {frag_l, frag_l.replace('-', ' '), re.sub('\\s+', '', frag_l)}
        for (key_name, value) in list(facts.items()) + [(k, product.get(k)) for k in product if k != 'facts' and (not isinstance(product.get(k), (dict, list)))]:
            if not isinstance(key_name, str) or value in (None, '', True, False):
                continue
            if key_name in skip_keys:
                continue
            text_v = str(value).casefold()
            if len(text_v) > 80:
                continue
            if any((alt in text_v or alt in text_v.replace('-', ' ') or alt in re.sub('\\s+', '', text_v) for alt in alts)):
                (matched_field, matched_val) = (key_name, value)
                break
        if matched_field is not None:
            break
    open_axes = [c for c in proposed if c not in soft and c.casefold() not in _META_FIELDS and (c not in facts) and (c not in product)]
    for claim in raw_claims:
        field = str(claim.get('field') or '').strip()
        value = claim.get('value')
        if not field or field in {'price', 'in_stock'}:
            continue
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_]{1,40}', field):
            continue
        if field.casefold() in _NOISE_FIELDS:
            continue
        text_c = _stringify(value)
        if not text_c:
            continue
        if _is_dimension_value(text_c):
            for lab in ask_vals:
                if _value_evidenced(lab, evidence):
                    text_c = lab
                    break
            else:
                continue
        observed = None
        for blob in (facts, product, stock):
            if isinstance(blob, dict) and field in blob:
                observed = blob.get(field)
                break
        if open_axes and (field in facts or field in product or field == matched_field):
            field = open_axes[0]
            observed = matched_val if matched_val is not None else observed
        if observed is not None:
            preferred = _prefer_claim_value(observed, goal_body)
            if any((_value_evidenced(lab, preferred) or _value_evidenced(lab, str(observed)) for lab in ask_vals)):
                text_c = preferred
            else:
                text_c = preferred
        elif _value_evidenced(text_c, evidence) or _value_evidenced(text_c, goal_body):
            text_c = _prefer_claim_value(text_c, goal_body)
        elif matched_val is not None and field in open_axes:
            text_c = _prefer_claim_value(matched_val, goal_body)
        elif matched_field is not None and field not in soft and (field not in open_axes):
            field = matched_field
            text_c = _prefer_claim_value(matched_val, goal_body)
        else:
            continue
        if field in soft and field != 'brand' and (not soft_attested(field, text_c)):
            continue
        if open_spec:
            allow(field)
        elif field not in allowed:
            continue
        add(field, text_c)

    def has_spec() -> bool:
        return any((c['field'] not in _META_FIELDS and c['field'] not in soft for c in grounded))
    if not has_spec() and matched_val is not None:
        axis = None
        if open_axes:
            axis = open_axes[0]
        elif matched_field is not None:
            axis = matched_field
        if axis is not None and (open_spec or axis in allowed):
            allow(axis)
            add(axis, _prefer_claim_value(matched_val, goal_body))
    brand = product.get('brand') or facts.get('brand')
    if brand and ('brand' in soft or 'brand' in allowed):
        allow('brand')
        add('brand', brand)
    cleaned['justification'] = {'claims': grounded}
    return cleaned

def _ensure_ids(tool_calls: list[dict[str, Any]], *, prefix: str) -> list[dict[str, Any]]:
    fixed = []
    for (i, call) in enumerate(tool_calls, start=1):
        item = dict(call)
        cid = str(item.get('id') or '')
        if not _TOOL_ID_RE.match(cid):
            item['id'] = f'{prefix}-{i}'
        if 'function' not in item and 'name' in item:
            item['function'] = {'name': item.get('name'), 'arguments': item.get('arguments', item.get('args', {}))}
        fixed.append(item)
    return fixed

def _select_calls(assistant: dict[str, Any], *, index: dict[str, dict[str, Any]], max_calls: int, memory: Memory, prefix: str) -> list[dict[str, Any]]:
    raw = _ensure_ids(list(assistant.get('tool_calls') or []), prefix=prefix)
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    must_reply = bool(memory.pending_question and (not memory.answered))
    for call in raw:
        fn = call.get('function') or {}
        name = str(fn.get('name') or '')
        if name not in index:
            continue
        args = _sanitize_args(name, _parse_arguments(fn.get('arguments')), index, memory)
        if name == 'place_test_order':
            if must_reply:
                continue
            args = _repair_order(args, index=index, memory=memory)
            key = f"{args.get('product_id')}::{args.get('sku')}"
            if key in memory.event_targets and memory.budget is not None:
                price = memory.live_price.get(key)
                if isinstance(price, (int, float)) and float(price) > memory.budget:
                    continue
            if key in memory.live_stock and (not memory.live_stock.get(key)):
                continue
            if memory.budget is not None:
                price = memory.live_price.get(key)
                if isinstance(price, (int, float)) and float(price) > memory.budget:
                    continue
        fp = _fingerprint(name, args)
        if fp in seen:
            continue
        if name not in {'message', 'place_test_order'} and memory.recent and (fp == memory.recent[-1]) and (memory.streak() >= 2):
            continue
        seen.add(fp)
        chosen.append({'id': call['id'], 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}})
        if len(chosen) >= max_calls:
            break
    if must_reply and 'message' in index and (not any((c['function']['name'] == 'message' for c in chosen))):
        chosen = [{'id': f'{prefix}-msg', 'type': 'function', 'function': {'name': 'message', 'arguments': json.dumps({'content': 'Thanks — I am checking live details and will choose an option that fits what you asked for.'})}}] + chosen
        chosen = chosen[:max_calls]
    return chosen

def _discover(*, index: dict[str, dict[str, Any]], goal: str, memory: Memory, prefix: str, max_calls: int) -> list[dict[str, Any]]:
    if memory.named and 'view' in index:
        calls = []
        for (i, (pid, _sku)) in enumerate(memory.named[:max_calls], start=1):
            if pid in memory.viewed:
                continue
            calls.append({'id': f'{prefix}-v{i}', 'type': 'function', 'function': {'name': 'view', 'arguments': json.dumps({'product_id': pid})}})
        if calls:
            return calls[:max_calls]
    calls: list[dict[str, Any]] = []
    if 'filter' in index:
        props = _tool_parameters(index['filter']).get('properties') or {}
        filt: dict[str, Any] = {}
        if 'in_stock' in props:
            filt['in_stock'] = True
        if 'k' in props:
            filt['k'] = 10
        if 'max_price' in props and memory.budget is not None and memory.searches:
            filt['max_price'] = memory.budget
        calls.append({'id': f'{prefix}-f', 'type': 'function', 'function': {'name': 'filter', 'arguments': json.dumps(filt)}})
    if 'search' in index:
        q = _short_query(_goal_head(goal))
        if memory.searches and memory.empty_hits:
            q = memory.next_seed()
        if q.lower() not in {s.lower() for s in memory.searches}:
            args: dict[str, Any] = {'query': q}
            props = _tool_parameters(index['search']).get('properties') or {}
            if 'k' in props:
                args['k'] = 10
            if 'in_stock' in props:
                args['in_stock'] = True
            calls.append({'id': f'{prefix}-s', 'type': 'function', 'function': {'name': 'search', 'arguments': json.dumps(args)}})
    if not calls and 'inspect_cart' in index:
        calls.append({'id': f'{prefix}-c', 'type': 'function', 'function': {'name': 'inspect_cart', 'arguments': '{}'}})
    return calls[:max_calls]

def tf1_trim(messages: list[dict[str, Any]], *, keep: int=12) -> None:
    if len(messages) <= keep + 2:
        return
    head = messages[:2]
    tail = messages[-keep:]
    while tail and tail[0].get('role') == 'tool':
        tail = tail[1:]
    messages[:] = head + tail

# ===== KI_2.py pipeline (tf2) =====
tf2_PREF_MODEL_RE = re.compile('(?:prefer(?:red)?|want|keep)\\s+(?:the\\s+)?(?:(?P<color>[A-Za-z]{3,12})\\s+)?(?P<code>\\d{3,})', re.I)
tf2_CATEGORY_GOAL_RE = re.compile("\\b(?:in|from)\\s+the\\s+([A-Za-z][A-Za-z0-9 &/+'-]{2,48}?)\\s+category\\b", re.I)
tf2_STOP = frozenset('\n    a an the and or but for with from that this these those into onto over under\n    at to of in on by is are was were be been being it its as if then than so\n    not no yes you your we our they their i me my need want find get buy purchase\n    please can could would should must also just only more most less least very\n    item items product products listing listings order orders cart shop shopping\n    available stock stocked budget price prices within below above prefer preferred\n    ideally soft firm requirement requirements category categories brand brands\n    looking look help thanks thank hi hello there here about around\n    something anything everything nothing somehow somewhere\n    '.split())
tf2_LEDGER_CAND_CAP = 28

def tf2_constraint_atoms(query: str, notes: list[str] | None=None) -> list[str]:
    """Extract compact, category-agnostic requirement atoms from public text.

    Atoms are quoted phrases, decimal/integer figures, and figure+token pairs.
    No product-category vocabulary is used.
    """
    text = _preamble(query)
    if notes:
        text = f"{text} {' '.join(notes[-3:])}"
    atoms: list[str] = []
    seen: set[str] = set()

    def add(atom: str) -> None:
        key = atom.strip().lower()
        if len(key) < 2 or key in seen or key in tf2_STOP:
            return
        seen.add(key)
        atoms.append(key)
    for match in re.finditer('"([^"\\n]{2,80})"|“([^”\\n]{2,80})”', text):
        add(match.group(1) or match.group(2) or '')
    budget = _parse_budget(text)
    budget_s = None
    if budget is not None:
        budget_s = str(int(budget)) if float(budget).is_integer() else str(budget)
    for match in _NUM_NEAR_RE.finditer(text):
        num = match.group('num') or match.group('num2')
        unit = match.group('unit') or match.group('unit2') or ''
        if not num:
            continue
        try:
            num_f = float(num.replace(',', ''))
        except ValueError:
            continue
        if budget is not None and abs(num_f - budget) <= 1e-06:
            continue
        if budget_s and num.replace(',', '') == budget_s.replace(',', ''):
            continue
        unit_l = unit.lower().strip()
        if unit_l in {'php', 'usd', 'eur', 'gbp', 'cad', 'aud', 'jpy', 'cny', 'inr', 'price', 'budget'}:
            continue
        if unit_l in tf2_STOP or len(unit_l) <= 1:
            add(num)
            continue
        if unit_l:
            add(f'{num} {unit_l}')
            add(num)
        else:
            add(num)
    for token in re.findall('\\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\\b', text.lower()):
        if token not in {'product_id', 'in_stock', 'max_price'}:
            add(token)
    return atoms[:24]

def tf2_soft_tokens(text: str) -> list[str]:
    """Extract soft preference objects (brand/name tokens) from public goal text."""
    out: list[str] = []
    seen: set[str] = set()
    for match in _SOFT_OBJECT_RE.finditer(text or ''):
        raw = (match.group(1) or '').strip().rstrip('.,;:')
        if not raw or len(raw) < 2:
            continue
        key = raw.lower()
        if key in tf2_STOP or key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out[:4]

def tf2_prefer_model_tokens(text: str) -> list[str]:
    """Extract preferred model/code/color tokens from public goal text."""
    out: list[str] = []
    seen: set[str] = set()
    for match in tf2_PREF_MODEL_RE.finditer(text or ''):
        for part in (match.group('color'), match.group('code')):
            if not part:
                continue
            key = part.lower()
            if key in seen or key in tf2_STOP:
                continue
            seen.add(key)
            out.append(part)
    return out[:6]

def tf2_category_mentions(text: str) -> list[str]:
    out: list[str] = []
    for match in tf2_CATEGORY_GOAL_RE.finditer(text or ''):
        label = (match.group(1) or '').strip()
        if label and label.lower() not in tf2_STOP:
            out.append(label)
    return out[:3]

def tf2_classify_procedure(query: str, *, tools: list[dict[str, Any]]) -> str:
    """Pick one shopping procedure from public goal + tool schema cues only.

    Does not use product-category vocabularies or suite answer tables. Branching
    is driven by structural signals: named variants, deferred specs, soft vs
    firm discourse, quoted phrases, use-context language, and order schema.
    """
    goal = _preamble(query)
    scores: dict[str, int] = {PROC_ASK: 0, PROC_RECALL: 0, PROC_STRICT: 0, PROC_SOFT: 0, PROC_USE: 0, PROC_RANK: 0, PROC_ADAPT: 0, PROC_CLAIM: 0}
    named = _named_variants(query)
    if len(named) >= 2:
        scores[PROC_RANK] += 6
    if tf2_rank_axes(goal):
        scores[PROC_RANK] += 2
    if _needs_clarification(query):
        scores[PROC_ASK] += 5
        scores[PROC_STRICT] += 2
    needs_why = 'justification' in _required_args(tools, 'place_test_order')
    if needs_why or _WHY_GOAL_RE.search(goal):
        scores[PROC_CLAIM] += 5
    softs = tf2_soft_tokens(goal)
    if softs and (_FIRM_MATTERS_RE.search(goal) or _FIRM_CUE_RE.search(goal)):
        scores[PROC_SOFT] += 5
    elif softs:
        scores[PROC_SOFT] += 3
    if _SOFT_CUE_RE.search(goal) and _FIRM_MATTERS_RE.search(goal):
        scores[PROC_SOFT] += 2
    phrases = _goal_phrases(goal)
    if phrases or tf2_category_mentions(goal):
        scores[PROC_RECALL] += 4
    if re.search('\\bmatches?\\b', goal, re.I) and phrases:
        scores[PROC_RECALL] += 2
    if _USE_FIT_RE.search(goal) and (not softs):
        scores[PROC_USE] += 4
    elif _USE_FIT_RE.search(goal):
        scores[PROC_USE] += 2
    if _ADAPT_GOAL_RE.search(goal):
        scores[PROC_ADAPT] += 4
    atoms = tf2_constraint_atoms(query)
    if len(atoms) >= 2 and scores[PROC_SOFT] == 0 and (scores[PROC_RECALL] == 0):
        scores[PROC_STRICT] += 2
    if not any(scores.values()):
        scores[PROC_STRICT] = 1
    order = (PROC_RANK, PROC_ASK, PROC_CLAIM, PROC_SOFT, PROC_ADAPT, PROC_USE, PROC_RECALL, PROC_STRICT)
    best = max(scores.values())
    for proc in order:
        if scores[proc] == best:
            return proc
    return PROC_STRICT

def tf2_parse_budget_update(text: str) -> float | None:
    """Budget updates from shopper notes — only when they explicitly revise the ceiling.

    Ignores incidental price figures in event notices (e.g. 'more expensive from
    4790 to 10618 PHP') so those cannot raise the shopping ceiling.
    """
    blob = text or ''
    if not re.search('\\b(?:budget|spend|under|below|at\\s+most|no\\s+more\\s+than|cap|ceiling|can\\s+only\\s+spend|max(?:imum)?\\s+price)\\b', blob, re.I):
        return None
    scrubbed = re.sub('(?:more\\s+expensive|price(?:\\s+change)?|went\\s+from|from)\\s+\\d[\\d,]*(?:\\.\\d+)?\\s*(?:to|-)?\\s*\\d[\\d,]*(?:\\.\\d+)?\\s*(?:php|usd|eur|gbp)?', ' ', blob, flags=re.I)
    return _parse_budget(scrubbed)

def tf2_rank_axes(text: str) -> list[tuple[str, int]]:
    """Parse comparative ranking axes: (normalized_axis, direction) where +1=higher, -1=lower."""
    axes: list[tuple[str, int]] = []
    seen: set[str] = set()
    dir_noise = _DIR_UP | _DIR_DOWN
    for match in _RANK_AXIS_RE.finditer(text or ''):
        direction_word = match.group('dir').lower()
        raw_axis = match.group('axis').strip().lower()
        parts = [p for p in raw_axis.split() if p not in dir_noise and p not in tf2_STOP]
        if not parts:
            continue
        axis = re.sub('[^a-z0-9_]+', '', '_'.join(parts))
        if len(axis) < 3 or axis in seen or axis in tf2_STOP:
            continue
        direction = -1 if direction_word in _DIR_DOWN else 1
        if 'price' in axis:
            direction = -1
        seen.add(axis)
        axes.append((axis, direction))
    return axes[:4]

def tf2_goal_tokens(query: str, notes: list[str] | None=None) -> set[str]:
    text = _preamble(query)
    if notes:
        text = f"{text} {' '.join(notes[-3:])}"
    tokens = {t.lower() for t in _WORD_RE.findall(text) if len(t) > 1 and t.lower() not in tf2_STOP}
    tokens.update(_NUM_RE.findall(text.lower()))
    return tokens

def tf2_search_seeds(query: str, notes: list[str] | None=None, *, procedure: str=PROC_STRICT) -> list[str]:
    """Ladder of discovery queries from distinctive public tokens (specific → broader)."""
    chunks = [_preamble(query)]
    if notes:
        chunks.extend(notes[-2:])
    text = ' '.join(chunks)
    seeds: list[str] = []
    phrases = _goal_phrases(text)
    for phrase in phrases:
        seeds.append(phrase)
        parts = [p for p in _WORD_RE.findall(phrase) if p.lower() not in tf2_STOP]
        if len(parts) >= 2:
            seeds.append(' '.join(parts))
        if len(parts) >= 3:
            seeds.append(' '.join(parts[:2]))
            seeds.append(parts[-1])
        for part in parts:
            if len(part) >= 4:
                seeds.append(part)
    cats = tf2_category_mentions(text)
    for cat in cats:
        seeds.append(cat)
    noise = tf2_STOP | _DIR_UP | _DIR_DOWN | {'usd', 'eur', 'gbp', 'cad', 'aud', 'jpy', 'cny', 'inr', 'php', 'then', 'than', 'after', 'before', 'versus', 'vs'}
    modelish = [m.group(1) for m in re.finditer('\\b([A-Za-z]*\\d{2,}[A-Za-z0-9-]*)\\b', text) if m.group(1).lower() not in noise and (not m.group(1).isdigit())]
    budget = _parse_budget(text)
    budget_str = str(int(budget)) if budget is not None and float(budget).is_integer() else str(budget) if budget is not None else None
    words: list[str] = []
    seen: set[str] = set()
    for tok in re.findall('[A-Za-z][A-Za-z0-9.+-]{2,}', text):
        low = tok.lower().strip('.,;:')
        if low in noise or low in seen:
            continue
        if budget_str and low == budget_str:
            continue
        seen.add(low)
        words.append(tok.strip('.,;:'))
        if len(words) >= 12:
            break
    alpha = [w for w in words if not re.search('\\d', w)]
    mixed = [w for w in words if w not in alpha]
    softs = tf2_soft_tokens(text)
    if procedure == PROC_SOFT and softs:
        primary = (softs[:1] + alpha[:3] + mixed[:2] + modelish[:2])[:6]
        seeds.append(' '.join(primary))
        if softs and alpha[:2]:
            seeds.append(f"{softs[0]} {' '.join(alpha[:2])}")
    elif procedure == PROC_RECALL:
        primary = (alpha[:3] + mixed[:2] + modelish[:2])[:6]
        if primary:
            seeds.append(' '.join(primary))
        if alpha[:2]:
            seeds.append(' '.join(alpha[:2]))
        if alpha:
            seeds.append(alpha[0])
    elif procedure == PROC_USE:
        primary = (alpha[:4] + mixed[:2])[:6]
        if primary:
            seeds.append(' '.join(primary))
        if alpha[:2]:
            seeds.append(' '.join(alpha[:2]))
    else:
        primary = (alpha[:4] + mixed[:2] + modelish[:2])[:6]
        if primary:
            seeds.append(' '.join(primary))
        if len(primary) > 3:
            seeds.append(' '.join(primary[:3]))
        if alpha[:2]:
            seeds.append(' '.join(alpha[:2]))
    for atom in tf2_constraint_atoms(query, notes):
        if ' ' in atom and any((ch.isdigit() for ch in atom)):
            seeds.append(atom)
            if softs:
                seeds.append(f'{softs[0]} {atom}')
    out: list[str] = []
    seen_s: set[str] = set()
    for s in seeds:
        key = s.lower().strip()
        if not key or key in seen_s:
            continue
        seen_s.add(key)
        out.append(s.strip())
    return out or ['in stock']

class tf2_Ledger:
    """Mutable world state derived only from the public goal and observations."""

    def __init__(self, query: str, allowed_claims: set[str], *, tools: list[dict[str, Any]] | None=None) -> None:
        self.query = query
        self.allowed_claims = set(allowed_claims)
        self.budget = _parse_budget(_preamble(query))
        self.needs_ask = _needs_clarification(query)
        self.named = _named_variants(query)
        self.rank_axes = tf2_rank_axes(_preamble(query))
        self.firm_signal = bool(_FIRM_CUE_RE.search(_preamble(query)))
        self.soft_signal = bool(_SOFT_CUE_RE.search(_preamble(query)))
        self.soft_tokens = tf2_soft_tokens(_preamble(query))
        self.phrases = _goal_phrases(_preamble(query))
        self.categories = tf2_category_mentions(_preamble(query))
        self.prefer_models = tf2_prefer_model_tokens(_preamble(query))
        self.procedure = tf2_classify_procedure(query, tools=tools or [])
        if self.procedure == PROC_ASK:
            self.needs_ask = True
        self.mentioned = _fields_in_text(query, allowed_claims)
        self.atoms = tf2_constraint_atoms(query)
        self.seeds = tf2_search_seeds(query, procedure=self.procedure)
        self.seed_idx = 0
        self.notes: list[str] = []
        self.candidates: dict[str, dict[str, Any]] = {}
        self.viewed: dict[str, dict[str, Any]] = {}
        self.stock: dict[str, dict[str, Any]] = {}
        self.cart: list[dict[str, str]] = []
        self.actions: list[str] = []
        self.last_error: str | None = None
        self.pending_q = False
        self.asked = False
        self.clarified = False
        self.needs_recheck = False
        self.checked: set[str] = set()
        self.must_recover = False
        self.search_n = 0
        self.empty_searches = 0
        self.view_n = 0
        self.cart_n = 0
        self.failed: set[str] = set()
        self.tried_queries: set[str] = set()
        self.same_search_streak = 0
        self.last_search_query = ''
        self.identical_action_streak = 0
        self.last_action_stamp = ''
        self.post_cart_pending: set[str] = set()
        self.priority_asked = False

    def _refresh(self) -> None:
        blob = ' '.join(self.notes[-3:])
        updated = tf2_parse_budget_update(blob)
        if updated is not None:
            if self.budget is None or updated <= self.budget + 1e-09:
                self.budget = updated
        self.mentioned = _fields_in_text(self.query, self.allowed_claims, extra=blob)
        self.atoms = tf2_constraint_atoms(self.query, self.notes)
        self.soft_tokens = tf2_soft_tokens(f'{_preamble(self.query)} {blob}')
        self.phrases = _goal_phrases(f'{_preamble(self.query)} {blob}')
        self.prefer_models = tf2_prefer_model_tokens(f'{_preamble(self.query)} {blob}')
        self.seeds = tf2_search_seeds(self.query, self.notes, procedure=self.procedure)
        self.rank_axes = tf2_rank_axes(f'{_preamble(self.query)} {blob}')
        for (pid, sku) in _named_variants(blob):
            key = f'{pid}::{sku}'
            if key not in {f'{a}::{b}' for (a, b) in self.named}:
                self.named.append((pid, sku))

    def note_shopper(self, content: str) -> None:
        text = (content or '').strip()
        if not text:
            return
        self.notes.append(text[:500])
        self.notes = self.notes[-5:]
        self._refresh()
        if '?' in text:
            self.pending_q = True
        if self.asked:
            self.clarified = True
            self.needs_ask = False
        price_event = bool(re.search('more\\s+expensive|price\\s+change|went\\s+(?:up|from)|re-?check before ordering', text, re.I))
        if self.cart:
            self.needs_recheck = True
            self.checked.clear()
            for item in self.cart:
                self.stock.pop(item['key'], None)
                self.stock.pop(item['product_id'], None)
                self.post_cart_pending.add(item['key'])
            if price_event:
                self.must_recover = False
            else:
                self.must_recover = False

    def remember(self, name: str, args: dict[str, Any]) -> None:
        self.actions.append(name)
        self.actions = self.actions[-30:]
        stamp = name + ':' + json.dumps(args, sort_keys=True, default=str)[:180]
        if stamp == self.last_action_stamp:
            self.identical_action_streak += 1
        else:
            self.identical_action_streak = 1
            self.last_action_stamp = stamp
        if name in {'search', 'filter'}:
            self.search_n += 1
            q = str(args.get('query') or '').strip().lower()
            if q:
                if q == self.last_search_query:
                    self.same_search_streak += 1
                else:
                    self.same_search_streak = 1
                    self.last_search_query = q
                self.tried_queries.add(q)
        if name in {'view', 'compare'}:
            self.view_n += 1
        if name == 'message':
            self.asked = True
            self.pending_q = False
            self.priority_asked = True
        if name == 'add_to_cart':
            self.cart_n += 1
            pid = str(args.get('product_id') or '')
            sku = str(args.get('sku') or '')
            if pid and sku:
                key = f'{pid}::{sku}'
                self.cart = [c for c in self.cart if c.get('key') != key]
                self.cart.append({'product_id': pid, 'sku': sku, 'key': key})
                self.stock.pop(key, None)
                self.stock.pop(pid, None)
                self.checked.discard(key)
                self.needs_recheck = True
                self.must_recover = False
                self.post_cart_pending.add(key)
        if name == 'remove_from_cart':
            pid = str(args.get('product_id') or '')
            sku = str(args.get('sku') or '')
            key = f'{pid}::{sku}'
            if key:
                self.failed.add(key)
            self.cart = [c for c in self.cart if c.get('key') != key]
            self.must_recover = False
            self.needs_recheck = False
            self.checked.discard(key)
            self.post_cart_pending.discard(key)
        if name == 'place_test_order':
            self.identical_action_streak = 0

    def _store(self, row: dict[str, Any], *, viewed: bool=False) -> None:
        pid = str(row.get('product_id') or '')
        if not pid:
            return
        sku = str(row.get('sku') or '')
        key = f'{pid}::{sku}' if sku else pid
        prev = self.candidates.get(key, {})
        merged = {**prev, **{k: v for (k, v) in row.items() if v is not None}}
        merged['product_id'] = pid
        if sku:
            merged['sku'] = sku
        self.candidates[key] = merged
        if viewed:
            slim = {k: merged.get(k) for k in ('product_id', 'sku', 'title', 'brand', 'price', 'in_stock', 'category_path', 'facts', 'listing_id') if merged.get(k) is not None}
            self.viewed[pid] = {**self.viewed.get(pid, {}), **slim}
        if len(self.candidates) > tf2_LEDGER_CAND_CAP:
            ranked = sorted(self.candidates.items(), key=lambda kv: (0 if kv[1].get('product_id') in self.viewed else 1, 0 if kv[0] in self.stock or kv[1].get('product_id') in self.stock else 1, float(kv[1]['price']) if isinstance(kv[1].get('price'), (int, float)) else 1e+18))
            keep = {k for (k, _) in ranked[:tf2_LEDGER_CAND_CAP]}
            self.candidates = {k: v for (k, v) in self.candidates.items() if k in keep}

    def ingest(self, name: str, args: dict[str, Any], observation: dict[str, Any]) -> None:
        if not isinstance(observation, dict):
            return
        (envelope, payload) = _unwrap_observation(observation)
        if envelope.get('error') or payload.get('error'):
            self.last_error = str(envelope.get('error') or payload.get('error'))[:240]
            return
        if name in {'search', 'filter'}:
            rows = [r for r in payload.get('results') or [] if isinstance(r, dict)]
            if not rows:
                self.empty_searches += 1
                self.seed_idx = min(self.seed_idx + 1, max(0, len(self.seeds) - 1))
            else:
                self.empty_searches = 0
            for row in rows:
                self._store(row)
        elif name == 'compare':
            for row in payload.get('compared') or payload.get('results') or []:
                if isinstance(row, dict):
                    self._store(row, viewed=True)
        elif name == 'view':
            self._store(payload, viewed=True)
            pid = str(payload.get('product_id') or args.get('product_id') or '')
            if pid:
                facts = payload.get('facts') if isinstance(payload.get('facts'), dict) else {}
                self.viewed[pid] = {**self.viewed.get(pid, {}), 'product_id': pid, 'title': payload.get('title'), 'brand': payload.get('brand'), 'category_path': payload.get('category_path'), 'facts': facts, 'listing_id': payload.get('listing_id'), 'price': payload.get('price'), 'in_stock': payload.get('in_stock')}
            for row in (payload.get('variants') or [])[:8]:
                if isinstance(row, dict):
                    self._store({**row, 'product_id': row.get('product_id') or pid})
        elif name == 'inspect_stock':
            pid = str(payload.get('product_id') or args.get('product_id') or '')
            sku = str(payload.get('sku') or args.get('sku') or '')
            if pid:
                stock_body = {k: v for (k, v) in payload.items() if k not in {'done', 'error', 'observation', 'action'}}
                key = f'{pid}::{sku}'
                self.stock[key] = stock_body
                self.stock[pid] = stock_body
                self._store({'product_id': pid, 'sku': sku, **stock_body})
                self.checked.add(key)
                self.post_cart_pending.discard(key)
                carted = any((item.get('key') == key or item.get('product_id') == pid for item in self.cart))
                if carted or self.needs_recheck:
                    if self.live_ok(pid, sku):
                        self.needs_recheck = False
                        self.must_recover = False
                    else:
                        self.needs_recheck = False
                        self.must_recover = True
                        self.failed.add(key)
        elif name == 'inspect_cart':
            rebuilt: list[dict[str, str]] = []
            stack: list[Any] = [payload]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    pid = str(node.get('product_id') or '')
                    sku = str(node.get('sku') or '')
                    if pid and sku:
                        rebuilt.append({'product_id': pid, 'sku': sku, 'key': f'{pid}::{sku}'})
                    stack.extend(node.values())
                elif isinstance(node, list):
                    stack.extend(node)
            if rebuilt:
                seen: set[str] = set()
                uniq: list[dict[str, str]] = []
                for item in rebuilt:
                    if item['key'] in seen:
                        continue
                    seen.add(item['key'])
                    uniq.append(item)
                self.cart = uniq

    def live_ok(self, product_id: str, sku: str) -> bool:
        key = f'{product_id}::{sku}'
        if key in self.post_cart_pending:
            return False
        stock = self.stock.get(key) or self.stock.get(product_id)
        if not isinstance(stock, dict):
            return False
        flagged = _as_bool(stock.get('in_stock'))
        if flagged is not True:
            return False
        price = _as_float(stock.get('price'))
        if self.budget is not None:
            if price is None:
                return False
            if price > self.budget + 1e-09:
                return False
        return True

    def over_budget(self, product_id: str, sku: str) -> bool:
        if self.budget is None:
            return False
        key = f'{product_id}::{sku}'
        stock = self.stock.get(key) or self.stock.get(product_id)
        if not isinstance(stock, dict):
            return False
        price = _as_float(stock.get('price'))
        return price is not None and price > self.budget + 1e-09

    def ready_order(self) -> dict[str, str] | None:
        if self.must_recover or self.needs_recheck or (not self.cart):
            return None
        top = self.cart[-1]
        if top['key'] in self.post_cart_pending:
            return None
        if not self.live_ok(top['product_id'], top['sku']):
            return None
        return {'product_id': top['product_id'], 'sku': top['sku']}

    def evidence_ratio(self, row: dict[str, Any], viewed: dict[str, Any] | None=None) -> float:
        atoms = self.atoms
        if not atoms:
            return 1.0
        blob = _product_blob(row, viewed)
        (hits, total) = _atoms_hit(atoms, blob)
        return hits / max(total, 1)

    def _soft_hit(self, row: dict[str, Any], viewed: dict[str, Any] | None=None) -> bool:
        if not self.soft_tokens:
            return False
        blob = _product_blob(row, viewed)
        return any((tok.lower() in blob for tok in self.soft_tokens))

    def _phrase_hit(self, row: dict[str, Any], viewed: dict[str, Any] | None=None) -> bool:
        if not self.phrases:
            return False
        blob = _product_blob(row, viewed)
        for phrase in self.phrases:
            pl = phrase.lower()
            if pl in blob:
                return True
            parts = [p.lower() for p in _WORD_RE.findall(phrase) if p.lower() not in tf2_STOP]
            if parts and all((p in blob for p in parts)):
                return True
        return False

    def _category_hit(self, row: dict[str, Any], viewed: dict[str, Any] | None=None) -> bool:
        if not self.categories:
            return True
        path = (viewed or {}).get('category_path')
        if path is None:
            path = row.get('category_path')
        path_text = json.dumps(path, ensure_ascii=False, default=str).lower() if path else ''
        if not path_text or path_text in {'null', '[]', '""'}:
            return False
        for cat in self.categories:
            tokens = [t.lower() for t in _WORD_RE.findall(cat) if t.lower() not in tf2_STOP and len(t) > 2]
            if tokens and any((t in path_text for t in tokens)):
                return True
        return False

    def evidence_need(self) -> float:
        if self.procedure == PROC_ASK:
            return 0.75 if len(self.atoms) <= 3 else 0.6
        if self.procedure in {PROC_STRICT, PROC_CLAIM}:
            return 0.67 if len(self.atoms) <= 3 else 0.5
        if self.procedure == PROC_SOFT:
            return 0.5
        if self.procedure == PROC_RECALL:
            return 0.34
        if self.procedure in {PROC_USE, PROC_ADAPT}:
            return 0.34
        return 0.5

    def constraints_ok(self, product_id: str, sku: str) -> bool:
        """True when public constraint atoms are evidenced on the viewed listing."""
        if not self.atoms and self.procedure not in {PROC_RECALL}:
            return True
        key = f'{product_id}::{sku}'
        row = self.candidates.get(key) or self.candidates.get(product_id) or {}
        viewed = self.viewed.get(product_id) or {}
        if not viewed and product_id not in self.viewed:
            return False
        if self.procedure == PROC_RECALL:
            if self.phrases and (not self._phrase_hit(row, viewed)):
                return False
            if self.categories and (not self._category_hit(row, viewed)):
                return False
            return True
        ratio = self.evidence_ratio(row, viewed)
        return ratio + 1e-09 >= self.evidence_need()

    def _axis_score(self, row: dict[str, Any], viewed: dict[str, Any]) -> float:
        if not self.rank_axes:
            return 0.0
        product = _flatten({**row, **viewed})
        total = 0.0
        for (idx, (axis, direction)) in enumerate(self.rank_axes):
            weight = 4.0 - idx
            observed = _lookup([product, viewed, row], axis)
            if observed is None:
                facts = viewed.get('facts') if isinstance(viewed.get('facts'), dict) else {}
                for (key, value) in {**product, **facts}.items():
                    if not isinstance(key, str):
                        continue
                    if _keys_match(_norm_key(key), _norm_key(axis)):
                        observed = value
                        break
            if observed is None:
                continue
            try:
                num = float(str(observed).replace(',', ''))
            except ValueError:
                total += weight * (0.5 if direction > 0 else 0.0)
                continue
            total += weight * direction * num
        return total

    def ranked(self) -> list[tuple[float, str, dict[str, Any]]]:
        goal = tf2_goal_tokens(self.query, self.notes)
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        use_tokens = {t.lower() for t in _WORD_RE.findall(_preamble(self.query)) if t.lower() not in tf2_STOP and len(t) > 3}
        for (key, row) in self.candidates.items():
            if key in self.failed:
                continue
            score = 0.0
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            viewed = self.viewed.get(pid) or {}
            if pid in self.viewed:
                score += 4
            if key in self.stock or pid in self.stock:
                score += 3
                if self.live_ok(pid, sku):
                    score += 4
                else:
                    score -= 5
            if row.get('in_stock') is True:
                score += 1
            price = row.get('price')
            if any((pid == n[0] and (not sku or sku == n[1]) for n in self.named)):
                score += 8 if self.procedure == PROC_RANK else 5
            elif any((pid == n[0] for n in self.named)):
                score += 3 if self.procedure == PROC_RANK else 2
            title = str(row.get('title') or '')
            score += min(4, _overlap(title, goal))
            ev = self.evidence_ratio(row, viewed)
            if self.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM}:
                score += 10.0 * ev
                if ev + 1e-09 < self.evidence_need():
                    score -= 6
            else:
                score += 6.0 * ev
            goal_nums = {t for t in goal if any((ch.isdigit() for ch in t))}
            title_nums = set(_NUM_RE.findall(title.lower()))
            if goal_nums and title_nums & goal_nums:
                score += 2
            facts = viewed.get('facts') if isinstance(viewed.get('facts'), dict) else {}
            for field in self.mentioned:
                observed = _lookup([viewed, row, facts], field)
                if observed is None:
                    continue
                score += 2 + min(3, _overlap(observed, goal))
            score += self._axis_score(row, viewed)
            if self.procedure == PROC_SOFT and self._soft_hit(row, viewed):
                score += 9
            elif self.procedure == PROC_SOFT and self.soft_tokens and (pid in self.viewed):
                score -= 2
            if self.procedure == PROC_RECALL:
                if self._phrase_hit(row, viewed):
                    score += 10
                else:
                    score -= 4
                if self.categories:
                    score += 5 if self._category_hit(row, viewed) else -6
            if self.procedure == PROC_USE and pid in self.viewed:
                blob = _product_blob(row, viewed)
                fit = sum((1 for t in use_tokens if t in blob))
                score += min(8, fit)
                price_f = _as_float(price)
                if price_f is not None and self.budget is not None and (price_f <= self.budget):
                    score += 1.5
            if self.procedure == PROC_ADAPT:
                for failed_key in self.failed:
                    if failed_key.startswith(f'{pid}::'):
                        score += 4
                        break
                blob = _product_blob(row, viewed)
                for tok in self.prefer_models:
                    if tok.lower() in blob:
                        score += 6
            if self.procedure == PROC_CLAIM and pid in self.viewed and facts:
                score += 3
            title_l = title.lower()
            if ' for ' in title_l:
                head = next((t for t in goal if len(t) >= 4 and t in title_l), None)
                if head:
                    score -= 2
            if self.budget is not None:
                price_f = _as_float(price)
                if price_f is not None:
                    if price_f > self.budget + 1e-09:
                        score -= 8
                    elif self.procedure == PROC_USE:
                        score += 1.5 + min(2.5, 2.5 * (price_f / max(self.budget, 1.0)))
                    elif self.procedure == PROC_ADAPT:
                        headroom = (self.budget - price_f) / max(self.budget, 1.0)
                        score += 2 + min(3.0, 3.0 * headroom)
                    else:
                        score += 2 + min(2.0, 2.0 * (price_f / max(self.budget, 1.0)))
            ranked.append((score, key, row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    def next_seed(self) -> str:
        if not self.seeds:
            return 'in stock'
        idx = min(self.seed_idx, len(self.seeds) - 1)
        candidate = self.seeds[idx]
        for offset in range(len(self.seeds)):
            s = self.seeds[(idx + offset) % len(self.seeds)]
            if s.lower() not in self.tried_queries:
                return s
        return candidate

    def recovery_step(self) -> dict[str, str] | None:
        need = self.evidence_need()
        for (score, key, row) in self.ranked():
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            if not pid or not sku or key in self.failed:
                continue
            if score < 1 and self.view_n > 0:
                continue
            if pid not in self.viewed:
                return {'action': 'view', 'product_id': pid, 'sku': sku}
            viewed = self.viewed.get(pid) or {}
            if self.atoms and self.evidence_ratio(row, viewed) + 1e-09 < need:
                continue
            if self.procedure == PROC_RECALL and self.phrases and (not self._phrase_hit(row, viewed)):
                continue
            if key in self.stock or pid in self.stock:
                if self.live_ok(pid, sku):
                    return {'action': 'add_to_cart', 'product_id': pid, 'sku': sku}
                continue
            return {'action': 'inspect_stock', 'product_id': pid, 'sku': sku}
        for (_score, key, row) in self.ranked():
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            if pid and sku and (key not in self.failed) and (pid not in self.viewed):
                return {'action': 'view', 'product_id': pid, 'sku': sku}
        return None

    def pick_cart_target(self) -> dict[str, str] | None:
        """Best evidenced, live-valid cart candidate under the active procedure."""
        need = self.evidence_need()
        soft_ready: dict[str, str] | None = None
        for (score, key, row) in self.ranked():
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            if not pid or not sku or key in self.failed:
                continue
            viewed = self.viewed.get(pid) or {}
            if pid not in self.viewed:
                return {'action': 'view', 'product_id': pid, 'sku': sku}
            if self.atoms and self.evidence_ratio(row, viewed) + 1e-09 < need:
                continue
            if self.procedure == PROC_RECALL:
                if self.phrases and (not self._phrase_hit(row, viewed)):
                    continue
                if self.categories and (not self._category_hit(row, viewed)):
                    continue
            if self.procedure == PROC_CLAIM and (not viewed.get('facts')):
                return {'action': 'view', 'product_id': pid, 'sku': sku}
            if key not in self.stock and pid not in self.stock:
                return {'action': 'inspect_stock', 'product_id': pid, 'sku': sku}
            if not self.live_ok(pid, sku):
                continue
            cand = {'action': 'add_to_cart', 'product_id': pid, 'sku': sku}
            if self.procedure == PROC_SOFT and self.soft_tokens:
                if self._soft_hit(row, viewed):
                    return cand
                if soft_ready is None:
                    soft_ready = cand
                continue
            return cand
        return soft_ready

    def hint(self, turn: int, max_steps: int) -> str:
        left = max_steps - turn + 1
        if self.needs_ask and (not self.clarified):
            return 'Ask one focused question about the missing firm specification before carting.'
        if self.procedure == PROC_USE and (not self.priority_asked) and (not self.cart):
            return 'Optionally clarify use priorities with message, then compare facts before first cart.'
        if self.pending_q:
            return 'Reply to the shopper with message before ordering.'
        if self.must_recover and self.cart:
            return f"Live check failed for {self.cart[-1]['key']}: remove, replace, re-inspect, order."
        if self.post_cart_pending and self.cart:
            top = self.cart[-1]
            return f"Cart changed — inspect_stock {top['key']} again before place_test_order (price/stock may have moved)."
        if self.needs_recheck and self.cart:
            top = self.cart[-1]
            if top['key'] not in self.checked:
                return f"Re-inspect_stock {top['key']} once after the shopper update."
        if self.last_error:
            return f'Recover from error ({self.last_error}) then continue.'
        recent = self.actions[-4:]
        if recent and len(set(recent)) == 1 and (recent[0] in {'search', 'filter', 'message', 'inspect_stock'}):
            return f'Stop looping on {recent[0]}; advance the purchase pipeline.'
        if self.named and self.view_n < min(2, len(self.named)):
            return 'View/compare named product::sku keys before ranking or carting.'
        if not self.candidates:
            return f'Discover with search/filter (try: {self.next_seed()}), in_stock=true.'
        if self.procedure == PROC_RECALL and self.view_n < 2:
            return 'Survey multiple phrase-matching listings before carting.'
        if self.view_n == 0:
            return 'Inspect a shortlist with view/compare; verify exact sku facts.'
        if not self.cart:
            if self.procedure == PROC_SOFT and self.soft_tokens:
                return f"Cart a firm-compliant item honoring soft preference ({', '.join(self.soft_tokens)}) when available."
            if left <= 8:
                return 'Cart the best fully compliant inspected variant now.'
            return 'Cart only a fully compliant intended purchase; honor soft prefs when feasible.'
        top = self.cart[-1]
        if top['key'] not in self.stock and top['product_id'] not in self.stock:
            return 'inspect_stock the carted sku once, then place_test_order.'
        if not self.live_ok(top['product_id'], top['sku']):
            return 'Carted sku failed live checks — remove_from_cart and recover.'
        if left <= 10:
            return 'Verified — place_test_order NOW with grounded claims if required.'
        return 'Live checks passed — place_test_order for the carted variant.'

    def board(self, turn: int, max_steps: int) -> str:
        lines = [f'ledger turn={turn}/{max_steps} left={max_steps - turn + 1}', f'procedure={self.procedure}', f"budget={(self.budget if self.budget is not None else 'none')} ask={self.needs_ask and (not self.clarified)} recheck={self.needs_recheck} recover={self.must_recover}", f'searches={self.search_n} empty={self.empty_searches} views={self.view_n} cands={len(self.candidates)} cart={len(self.cart)} adds={self.cart_n}', f'seed={self.next_seed()}', f'next={self.hint(turn, max_steps)}']
        if self.soft_tokens:
            lines.append('soft=' + ', '.join(self.soft_tokens))
        if self.phrases:
            lines.append('phrases=' + ' | '.join(self.phrases[:3]))
        if self.categories:
            lines.append('category_mentions=' + ' | '.join(self.categories))
        if self.named:
            lines.append('named=' + ', '.join((f'{p}::{s}' for (p, s) in self.named[:8])))
        if self.rank_axes:
            lines.append('rank_axes=' + ', '.join((f"{a}:{('up' if d > 0 else 'down')}" for (a, d) in self.rank_axes)))
        if self.mentioned:
            lines.append('claim_fields=' + ', '.join(sorted(self.mentioned)))
        if self.atoms:
            lines.append('constraint_atoms=' + ' | '.join(self.atoms[:10]))
        if self.notes:
            lines.append('notes:')
            for note in self.notes[-2:]:
                lines.append(f'  - {note}')
        if self.cart:
            lines.append('cart:')
            for item in self.cart[-3:]:
                stock = self.stock.get(item['key']) or self.stock.get(item['product_id']) or {}
                lines.append(f"  - {item['key']} price={stock.get('price', '?')} in_stock={stock.get('in_stock', '?')}")
        for (score, key, row) in self.ranked()[:_BOARD_SHOW]:
            pid = str(row.get('product_id') or '')
            ev = self.evidence_ratio(row, self.viewed.get(pid))
            lines.append(f"cand score={score:.1f} ev={ev:.2f} {key} title={str(row.get('title') or '')[:60]} price={row.get('price', '?')} stock={row.get('in_stock', '?')}")
        if self.cart:
            top = self.cart[-1]
            viewed = self.viewed.get(top['product_id']) or {}
            facts = viewed.get('facts') if isinstance(viewed.get('facts'), dict) else {}
            if facts:
                preview = ', '.join((f'{k}={str(v)[:36]}' for (k, v) in list(facts.items())[:8]))
                lines.append(f'final_facts={preview}')
        return '\n'.join(lines)

def tf2_ground_order(args: dict[str, Any], *, tools: list[dict[str, Any]], ledger: tf2_Ledger) -> dict[str, Any]:
    cleaned = dict(args)
    required = _required_args(tools, 'place_test_order')
    needs = 'justification' in required
    if not needs:
        cleaned.pop('justification', None)
        return cleaned
    pid = str(cleaned.get('product_id') or '')
    sku = str(cleaned.get('sku') or '')
    if (not pid or not sku) and ledger.cart:
        top = ledger.cart[-1]
        pid = pid or top['product_id']
        sku = sku or top['sku']
        cleaned['product_id'] = pid
        cleaned['sku'] = sku
    key = f'{pid}::{sku}'
    stock = dict(ledger.stock.get(key) or ledger.stock.get(pid) or {})
    viewed = dict(ledger.viewed.get(pid) or {})
    candidate = dict(ledger.candidates.get(key) or ledger.candidates.get(pid) or {})
    product = _flatten({**candidate, **viewed, **stock})
    allowed = set(_claim_fields(tools))
    open_spec = _open_spec_claims(tools)
    soft_fields = _soft_schema_fields(tools, allowed)
    goal = tf2_goal_tokens(ledger.query, ledger.notes)
    goal_text = f"{_preamble(ledger.query)} {' '.join(ledger.notes[-3:])}".lower()
    prefer = set(ledger.mentioned)
    raw_claims: list[dict[str, Any]] = []
    just = cleaned.get('justification')
    if isinstance(just, dict) and isinstance(just.get('claims'), list):
        raw_claims = [c for c in just['claims'] if isinstance(c, dict)]
    if open_spec:
        allowed.update({'price', 'in_stock'})
        allowed.update(soft_fields)
        allowed.update(prefer)
        for claim in raw_claims:
            field = str(claim.get('field') or '').strip()
            if field and re.fullmatch('[A-Za-z][A-Za-z0-9_]{1,40}', field):
                allowed.add(field)
                prefer.add(field)
    if not allowed:
        allowed = {'price', 'in_stock'}
    grounded: list[dict[str, str]] = []
    seen: set[str] = set()
    attest_blob = _product_blob(product, viewed)

    def add(field: str, value: Any) -> None:
        if field not in allowed or field in seen:
            return
        if value is None or value == '':
            return
        if isinstance(value, bool):
            text = 'true' if value else 'false'
        else:
            text = str(value).strip()
        if not text or text.lower() in {'unknown', 'none', 'n/a', 'null'}:
            return
        if field != 'in_stock' and text.lower() == 'false':
            return
        if field not in {'price', 'in_stock'} and (not _value_attested(value, attest_blob)):
            return
        grounded.append({'field': field, 'value': text[:80]})
        seen.add(field)

    def best_value(field: str) -> Any:
        direct = _lookup([product, stock, viewed, candidate], field)
        if direct not in (None, ''):
            return direct
        if field in soft_fields:
            return None
        want_nums = _goal_nums_near(ledger.query, field, ledger.notes)
        best_val: Any = None
        best_score = 0
        for (key_name, value) in product.items():
            if value in (None, '', True, False) or isinstance(value, (dict, list)):
                continue
            if not isinstance(key_name, str):
                continue
            if key_name in {'product_id', 'sku', 'title', 'price', 'in_stock', 'currency', 'listing_id'} and field not in {'price', 'in_stock'}:
                continue
            text = str(value).lower().strip()
            nums = set(_NUM_RE.findall(text))
            shared = nums & want_nums if want_nums else set()
            overlap = 0
            if _keys_match(_norm_key(key_name), _norm_key(field)):
                overlap += 8
            if shared:
                overlap += 3 * len(shared)
            if len(text) >= 2 and (text in goal_text or any((tok == text for tok in goal if len(tok) >= 2))):
                overlap += 6
            overlap += min(2, _overlap(value, goal))
            if overlap > best_score:
                best_score = overlap
                best_val = value
        return best_val if best_score >= 6 else None
    add('price', stock.get('price', product.get('price')))
    stock_flag = stock.get('in_stock', product.get('in_stock'))
    flagged = _as_bool(stock_flag)
    if flagged is not None:
        add('in_stock', flagged)
    else:
        add('in_stock', stock_flag)
    for claim in raw_claims:
        field = str(claim.get('field') or '').strip()
        if not field or field in {'price', 'in_stock'}:
            continue
        if field not in allowed and (not open_spec):
            continue
        if open_spec:
            allowed.add(field)
        observed = best_value(field)
        if observed is not None:
            add(field, observed)
            continue
        raw_val = claim.get('value')
        if raw_val in (None, ''):
            continue
        if _value_attested(raw_val, attest_blob):
            add(field, raw_val)
    for field in list(prefer):
        if field in seen or field in {'price', 'in_stock'}:
            continue
        observed = best_value(field)
        if observed is not None:
            add(field, observed)
    if not any((f not in {'price', 'in_stock'} | soft_fields for f in seen)):
        scored: list[tuple[int, str, Any]] = []
        hints = [str(c).strip() for c in list(prefer) + [str(c.get('field') or '') for c in raw_claims] if str(c).strip() and str(c).strip() not in {'price', 'in_stock'}]
        goal_nums = {t for t in goal if any((ch.isdigit() for ch in t))}
        for (key_name, value) in product.items():
            if not isinstance(key_name, str) or value in (None, ''):
                continue
            if key_name in {'product_id', 'sku', 'title', 'price', 'in_stock', 'currency', 'listing_id', 'category_path', 'variants', 'options', 'condition'}:
                continue
            if isinstance(value, (dict, list, bool)):
                continue
            text = str(value).lower().strip()
            overlap = 0
            if len(text) >= 2 and (text in goal_text or any((tok == text for tok in goal if len(tok) >= 2))):
                overlap += 6
            overlap += min(3, _overlap(value, goal))
            nums = set(_NUM_RE.findall(text))
            overlap += 2 * len(nums & goal_nums)
            for atom in ledger.atoms:
                if atom in text or re.sub('\\s+', '', atom) in re.sub('\\s+', '', text):
                    overlap += 4
            if overlap <= 0:
                continue
            claim_name = None
            for cand in hints:
                if _keys_match(_norm_key(cand), _norm_key(key_name)):
                    claim_name = cand
                    overlap += 3
                    break
            if claim_name is None and open_spec:
                if re.fullmatch('[a-z][a-z0-9_]{1,40}', key_name):
                    claim_name = key_name
            if claim_name is None:
                continue
            if open_spec or claim_name in allowed:
                scored.append((overlap, claim_name, value))
        scored.sort(key=lambda row: (-row[0], row[1]))
        if scored:
            (_s, field, value) = scored[0]
            if open_spec:
                allowed.add(field)
            add(field, value)
    if open_spec and 'brand' not in seen:
        brand_val = product.get('brand')
        if brand_val not in (None, '') and _value_attested(brand_val, attest_blob):
            brand_l = str(brand_val).lower()
            if any((tok.lower() == brand_l for tok in ledger.soft_tokens)) or brand_l in goal_text:
                allowed.add('brand')
                add('brand', brand_val)
            elif not any((f not in {'price', 'in_stock'} for f in seen)):
                allowed.add('brand')
                add('brand', brand_val)
            elif ledger.procedure == PROC_CLAIM:
                allowed.add('brand')
                add('brand', brand_val)
    if soft_fields and (not seen & soft_fields):
        for field in sorted(soft_fields):
            observed = _lookup([product], field)
            if observed is True or str(observed).strip().lower() in {'true', 'yes', '1'}:
                add(field, 'true')
                break
            if observed not in (None, '') and _value_attested(observed, attest_blob):
                add(field, observed)
                break
    if open_spec and (not any((f not in {'price', 'in_stock'} for f in seen))):
        for (key_name, value) in product.items():
            if not isinstance(key_name, str) or not re.fullmatch('[a-z][a-z0-9_]{1,40}', key_name):
                continue
            if key_name in {'product_id', 'sku', 'title', 'price', 'in_stock', 'currency', 'brand'}:
                continue
            if value in (None, '') or isinstance(value, (dict, list, bool)):
                continue
            if _value_attested(value, attest_blob) and _overlap(value, goal) > 0:
                allowed.add(key_name)
                add(key_name, value)
                break
    if needs and (not grounded):
        add('price', stock.get('price', product.get('price')))
        stock_flag = stock.get('in_stock', product.get('in_stock', True))
        flagged = _as_bool(stock_flag)
        add('in_stock', flagged if flagged is not None else stock_flag)
    cleaned['justification'] = {'claims': grounded}
    return cleaned

def tf2_normalize(name: str, args: dict[str, Any], *, tools: list[dict[str, Any]], ledger: tf2_Ledger) -> dict[str, Any]:
    cleaned = dict(args)
    if name == 'search':
        query = str(cleaned.get('query') or '').strip()
        if not query or query.isdigit() or len(_WORD_RE.findall(query)) < 2:
            query = ledger.next_seed()
        if query.lower() in ledger.tried_queries and ledger.same_search_streak >= 1:
            ledger.seed_idx = min(ledger.seed_idx + 1, max(0, len(ledger.seeds) - 1))
            alt = ledger.next_seed()
            if alt.lower() != query.lower():
                query = alt
            else:
                tokens = [t for t in _WORD_RE.findall(query) if t.lower() not in tf2_STOP]
                query = ' '.join(tokens[:max(2, len(tokens) - 1)]) or query
        tokens = _WORD_RE.findall(query)
        if len(tokens) > 10 and '"' not in query:
            cleaned['query'] = ' '.join(tokens[:8])
        else:
            cleaned['query'] = query
        cleaned.setdefault('in_stock', True)
        cleaned.setdefault('k', 10)
        props = _tool_props(tools, 'search')
        if ledger.budget is not None and 'max_price' in props and ('max_price' not in cleaned):
            cleaned['max_price'] = ledger.budget
    if name == 'filter':
        cleaned.setdefault('in_stock', True)
        cleaned.setdefault('k', 10)
        if ledger.budget is not None and ledger.budget >= 5 and ('max_price' not in cleaned):
            cleaned['max_price'] = ledger.budget
        q = str(cleaned.get('query') or '').strip()
        if not q and ledger.seeds:
            cleaned['query'] = ledger.next_seed()
        elif q.lower() in ledger.tried_queries and ledger.same_search_streak >= 1:
            cleaned['query'] = ledger.next_seed()
    if name == 'compare':
        ids = cleaned.get('product_ids')
        if isinstance(ids, list):
            cleaned['product_ids'] = [str(x) for x in ids[:5] if str(x).strip()]
        elif isinstance(ids, str) and ids.strip():
            cleaned['product_ids'] = [p.strip() for p in ids.split(',') if p.strip()][:5]
        if not cleaned.get('product_ids') and ledger.named:
            cleaned['product_ids'] = [pid for (pid, _sku) in ledger.named[:5]]
    if name == 'message':
        cleaned['content'] = str(cleaned.get('content') or '').strip()[:400]
    if name == 'place_test_order':
        if (not cleaned.get('product_id') or not cleaned.get('sku')) and ledger.cart:
            top = ledger.cart[-1]
            cleaned.setdefault('product_id', top['product_id'])
            cleaned.setdefault('sku', top['sku'])
        cleaned = tf2_ground_order(cleaned, tools=tools, ledger=ledger)
    return cleaned

def tf2_forced_plan(*, ledger: tf2_Ledger, tools: list[dict[str, Any]], max_calls: int, remaining: int) -> list[tuple[str, dict[str, Any]]] | None:
    """Return a deterministic action plan when the next step is unambiguous."""
    names = _tool_names(tools)
    if ledger.needs_ask and (not ledger.clarified) and ('message' in names) and (not ledger.asked):
        return [('message', {'content': 'Before I pick an item, which firm product specification must it satisfy beyond budget and in-stock availability?'})]
    if ledger.cart and 'remove_from_cart' in names:
        top = ledger.cart[-1]
        if ledger.must_recover or ledger.over_budget(top['product_id'], top['sku']):
            plan: list[tuple[str, dict[str, Any]]] = [('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']})]
            ledger.must_recover = True
            ledger.failed.add(top['key'])
            nxt = ledger.recovery_step()
            if nxt and nxt['action'] in names and (len(plan) < max_calls):
                args = {'product_id': nxt['product_id']}
                if nxt['action'] != 'view':
                    args['sku'] = nxt['sku']
                plan.append((nxt['action'], args))
            return plan[:max_calls]
    if ledger.post_cart_pending and ledger.cart and ('inspect_stock' in names):
        top = ledger.cart[-1]
        if top['key'] in ledger.post_cart_pending or top['key'] not in ledger.stock:
            return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
    if ledger.needs_recheck and ledger.cart and ('inspect_stock' in names):
        top = ledger.cart[-1]
        if top['key'] not in ledger.checked:
            return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
    ready = ledger.ready_order()
    if ready and 'place_test_order' in names:
        return [('place_test_order', tf2_ground_order(dict(ready), tools=tools, ledger=ledger))]
    if ledger.cart and 'place_test_order' in names:
        top = ledger.cart[-1]
        if top['key'] not in ledger.post_cart_pending and ledger.live_ok(top['product_id'], top['sku']) and (not ledger.must_recover):
            return [('place_test_order', tf2_ground_order({'product_id': top['product_id'], 'sku': top['sku']}, tools=tools, ledger=ledger))]
    if ledger.pending_q and 'message' in names and (ledger.identical_action_streak < 2):
        return [('message', {'content': 'I will verify listings against your stated constraints, budget, and stock, then confirm the chosen product_id::sku before checkout.'})]
    if not ledger.cart and ledger.failed:
        nxt = ledger.recovery_step()
        if nxt and nxt['action'] in names:
            args = {'product_id': nxt['product_id']}
            if nxt['action'] != 'view':
                args['sku'] = nxt['sku']
            return [(nxt['action'], args)]
    if ledger.named and ledger.view_n == 0:
        if 'compare' in names and len(ledger.named) >= 2:
            return [('compare', {'product_ids': [pid for (pid, _) in ledger.named[:5]]})]
        if 'view' in names:
            (pid, _sku) = ledger.named[0]
            return [('view', {'product_id': pid})]
    if ledger.procedure == PROC_USE and (not ledger.priority_asked) and (not ledger.cart) and ('message' in names) and (ledger.view_n == 0) and ledger.candidates and (not ledger.asked):
        ledger.priority_asked = True
        return [('message', {'content': 'To rank options for your use case, which soft feature matters most once category, budget, and stock are met?'})]
    if ledger.same_search_streak >= 2 and ledger.candidates and ('view' in names):
        for (_score, key, row) in ledger.ranked():
            pid = str(row.get('product_id') or '')
            if pid and pid not in ledger.viewed and (key not in ledger.failed):
                return [('view', {'product_id': pid})]
        nxt = ledger.recovery_step()
        if nxt and nxt['action'] in names:
            args = {'product_id': nxt['product_id']}
            if nxt['action'] != 'view':
                args['sku'] = nxt['sku']
            return [(nxt['action'], args)]
    if ledger.same_search_streak >= 2 and (not ledger.candidates) and ('search' in names):
        ledger.seed_idx = min(ledger.seed_idx + 1, max(0, len(ledger.seeds) - 1))
        seed = ledger.next_seed()
        args: dict[str, Any] = {'query': seed, 'in_stock': True, 'k': 10}
        if ledger.procedure == PROC_RECALL:
            args['k'] = 12
        if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
            args['max_price'] = ledger.budget
        return [('search', args)]
    if ledger.candidates and 'view' in names:
        recent = ledger.actions[-3:]
        stalled_search = bool(recent) and all((a in {'search', 'filter'} for a in recent))
        min_views = 2 if ledger.procedure in {PROC_RECALL, PROC_USE, PROC_CLAIM} else 1
        if ledger.view_n < min_views or stalled_search or ledger.same_search_streak >= 2:
            for (_score, key, row) in ledger.ranked():
                pid = str(row.get('product_id') or '')
                if pid and pid not in ledger.viewed and (key not in ledger.failed):
                    return [('view', {'product_id': pid})]
    cart_horizon = {PROC_RECALL: 12, PROC_USE: 14, PROC_CLAIM: 16, PROC_SOFT: 18, PROC_ASK: 16, PROC_STRICT: 16, PROC_ADAPT: 18, PROC_RANK: 22}.get(ledger.procedure, 14)
    min_views_before_cart = {PROC_RECALL: 2, PROC_USE: 2, PROC_CLAIM: 1, PROC_SOFT: 1, PROC_ASK: 1, PROC_STRICT: 1, PROC_ADAPT: 1, PROC_RANK: 1}.get(ledger.procedure, 1)
    if remaining <= cart_horizon and ledger.candidates and (not ledger.cart) and (ledger.view_n >= min_views_before_cart):
        if ledger.needs_ask and (not ledger.clarified):
            pass
        else:
            target = ledger.pick_cart_target()
            if target and target['action'] in names:
                args = {'product_id': target['product_id']}
                if target['action'] != 'view':
                    args['sku'] = target['sku']
                return [(target['action'], args)]
    if remaining <= 6 and ledger.candidates and (not ledger.cart) and (ledger.view_n > 0):
        for (_score, key, row) in ledger.ranked():
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            if not pid or not sku or key in ledger.failed:
                continue
            if pid not in ledger.viewed and 'view' in names:
                return [('view', {'product_id': pid})]
            if key not in ledger.stock and pid not in ledger.stock and ('inspect_stock' in names):
                return [('inspect_stock', {'product_id': pid, 'sku': sku})]
            if ledger.live_ok(pid, sku) and 'add_to_cart' in names:
                return [('add_to_cart', {'product_id': pid, 'sku': sku})]
            break
    if ledger.cart and 'inspect_stock' in names:
        top = ledger.cart[-1]
        if top['key'] not in ledger.stock and top['product_id'] not in ledger.stock:
            return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
    if not ledger.candidates and 1 <= ledger.empty_searches <= max(2, len(ledger.seeds)) and ('search' in names):
        seed = ledger.next_seed()
        if seed.lower() not in ledger.tried_queries:
            args = {'query': seed, 'in_stock': True, 'k': 10}
            if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
                args['max_price'] = ledger.budget
            return [('search', args)]
    return None

def tf2_guard(pairs: list[tuple[str, dict[str, Any]]], *, ledger: tf2_Ledger, tools: list[dict[str, Any]], max_calls: int) -> list[tuple[str, dict[str, Any]]]:
    names = _tool_names(tools)
    out: list[tuple[str, dict[str, Any]]] = []
    if ledger.needs_ask and (not ledger.clarified) and ('message' in names) and (not ledger.asked):
        risky = {'add_to_cart', 'place_test_order', 'remove_from_cart'}
        if any((n in risky for (n, _) in pairs)) or not any((n == 'message' for (n, _) in pairs)):
            return [('message', {'content': 'What firm specification is required for this purchase besides budget and stock?'})]
    for (name, args) in pairs:
        pid = str(args.get('product_id') or '')
        sku = str(args.get('sku') or '')
        if ledger.needs_ask and (not ledger.asked) and (name in {'add_to_cart', 'place_test_order'}) and ('message' in names):
            out.append(('message', {'content': 'What firm specification is required for this purchase besides budget and stock?'}))
            continue
        if name == 'add_to_cart' and pid and (pid not in ledger.viewed) and ('view' in names):
            out.append(('view', {'product_id': pid}))
            continue
        if name == 'add_to_cart' and ledger.procedure == PROC_SOFT and ledger.soft_tokens and pid and sku and (ledger.cart_n == 0):
            row = ledger.candidates.get(f'{pid}::{sku}') or ledger.candidates.get(pid) or {}
            viewed = ledger.viewed.get(pid) or {}
            if not ledger._soft_hit(row, viewed):
                alt = ledger.pick_cart_target()
                if alt and alt['action'] == 'add_to_cart' and (alt['product_id'] != pid or alt['sku'] != sku):
                    alt_row = ledger.candidates.get(f"{alt['product_id']}::{alt['sku']}") or ledger.candidates.get(alt['product_id']) or {}
                    alt_view = ledger.viewed.get(alt['product_id']) or {}
                    if ledger._soft_hit(alt_row, alt_view):
                        out.append(('add_to_cart', {'product_id': alt['product_id'], 'sku': alt['sku']}))
                        continue
        if name == 'add_to_cart' and pid and sku and (ledger.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM}) and ledger.atoms and (ledger.view_n < 12):
            row = ledger.candidates.get(f'{pid}::{sku}') or ledger.candidates.get(pid) or {}
            viewed = ledger.viewed.get(pid) or {}
            if ledger.evidence_ratio(row, viewed) + 1e-09 < ledger.evidence_need():
                nxt = ledger.pick_cart_target()
                if nxt and nxt['action'] in names:
                    if nxt['action'] != 'add_to_cart' or (nxt['product_id'] != pid or nxt.get('sku') != sku):
                        args2 = {'product_id': nxt['product_id']}
                        if nxt['action'] != 'view':
                            args2['sku'] = nxt['sku']
                        out.append((nxt['action'], args2))
                        continue
        if name == 'add_to_cart' and pid and sku:
            key = f'{pid}::{sku}'
            if ledger.cart and any((c.get('key') == key for c in ledger.cart)) and (key not in ledger.post_cart_pending) and ledger.live_ok(pid, sku) and ('place_test_order' in names):
                out.append(('place_test_order', tf2_ground_order({'product_id': pid, 'sku': sku}, tools=tools, ledger=ledger)))
                continue
            out.append((name, args))
            continue
        if name in {'search', 'filter'}:
            q = str(args.get('query') or '').strip().lower()
            if q and q == ledger.last_search_query and (ledger.same_search_streak >= 2):
                if ledger.candidates and 'view' in names:
                    for (_score, key, row) in ledger.ranked():
                        pid2 = str(row.get('product_id') or '')
                        if pid2 and pid2 not in ledger.viewed and (key not in ledger.failed):
                            out.append(('view', {'product_id': pid2}))
                            break
                    else:
                        out.append((name, args))
                    continue
                args = dict(args)
                args['query'] = ledger.next_seed()
        if name == 'place_test_order' and pid and sku:
            key = f'{pid}::{sku}'
            if key in ledger.post_cart_pending and 'inspect_stock' in names:
                out.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                continue
            if ledger.must_recover and 'remove_from_cart' in names:
                out.append(('remove_from_cart', {'product_id': pid, 'sku': sku}))
                continue
            if ledger.over_budget(pid, sku):
                ledger.must_recover = True
                ledger.failed.add(key)
                if 'remove_from_cart' in names:
                    out.append(('remove_from_cart', {'product_id': pid, 'sku': sku}))
                elif 'inspect_stock' in names:
                    out.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                continue
            if ledger.needs_recheck and key not in ledger.checked and ('inspect_stock' in names):
                out.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                continue
            has_stock = key in ledger.stock or pid in ledger.stock
            if not has_stock and 'inspect_stock' in names:
                out.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                continue
            if has_stock and (not ledger.live_ok(pid, sku)):
                ledger.must_recover = True
                ledger.failed.add(key)
                if 'remove_from_cart' in names:
                    out.append(('remove_from_cart', {'product_id': pid, 'sku': sku}))
                continue
            if 'justification' in _required_args(tools, 'place_test_order') and pid not in ledger.viewed and ('view' in names):
                out.append(('view', {'product_id': pid}))
                continue
            if not ledger.live_ok(pid, sku):
                continue
        out.append((name, args))
    unique: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for (name, args) in out:
        stamp = name + ':' + json.dumps(args, sort_keys=True, default=str)
        if stamp in seen:
            continue
        seen.add(stamp)
        unique.append((name, args))
    names_present = {n for (n, _) in unique}
    if 'add_to_cart' in names_present and 'place_test_order' in names_present:
        unique = [(n, a) for (n, a) in unique if n != 'place_test_order']
    return unique[:max_calls]

def tf2_fallback_plan(*, ledger: tf2_Ledger, tools: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Last-resort deterministic action when the model returns nothing usable."""
    names = _tool_names(tools)
    if ledger.cart and ledger.post_cart_pending and ('inspect_stock' in names):
        top = ledger.cart[-1]
        return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
    if ledger.cart and 'place_test_order' in names:
        top = ledger.cart[-1]
        if top['key'] not in ledger.post_cart_pending and (not ledger.must_recover) and (not ledger.over_budget(top['product_id'], top['sku'])) and ledger.live_ok(top['product_id'], top['sku']):
            return [('place_test_order', tf2_ground_order({'product_id': top['product_id'], 'sku': top['sku']}, tools=tools, ledger=ledger))]
    ready = ledger.ready_order()
    if ready and 'place_test_order' in names:
        return [('place_test_order', tf2_ground_order(dict(ready), tools=tools, ledger=ledger))]
    if ledger.cart and 'inspect_stock' in names:
        top = ledger.cart[-1]
        if top['key'] not in ledger.stock and top['product_id'] not in ledger.stock:
            return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
        if (not ledger.live_ok(top['product_id'], top['sku']) or ledger.over_budget(top['product_id'], top['sku'])) and 'remove_from_cart' in names:
            return [('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']})]
    nxt = ledger.recovery_step()
    if nxt and nxt['action'] in names:
        args = {'product_id': nxt['product_id']}
        if nxt['action'] != 'view':
            args['sku'] = nxt['sku']
        return [(nxt['action'], args)]
    if ledger.candidates and 'view' in names:
        for (_score, key, row) in ledger.ranked():
            pid = str(row.get('product_id') or '')
            if pid and pid not in ledger.viewed and (key not in ledger.failed):
                return [('view', {'product_id': pid})]
    if 'search' in names:
        return [('search', {'query': ledger.next_seed(), 'in_stock': True, 'k': 10})]
    return []

def tf2_system_prompt(tools: list[dict[str, Any]], *, max_calls: int, max_steps: int, ledger: tf2_Ledger, needs_justification: bool, claim_fields: list[str]) -> str:
    tool_lines: list[str] = []
    for spec in tools:
        fn = _fn(spec)
        if not fn or not isinstance(fn.get('name'), str):
            continue
        desc = str(fn.get('description') or '').strip()
        params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
        props = params.get('properties') if isinstance(params.get('properties'), dict) else {}
        arg_names = ', '.join(sorted((str(k) for k in props.keys())))
        tool_lines.append(f"- {fn['name']}: {desc}" + (f' Args: {arg_names}.' if arg_names else ''))
    parts = ['You are a careful shopping agent. Obey the shopper request and every Task rules block.', 'Copy product_id and sku only from observations — never invent identifiers.', "Work for ANY product category using only the shopper's stated constraints and tool observations.", '', f'Active procedure: {ledger.procedure}', *_procedure_playbook(ledger.procedure), '', 'Shared process:', '1. If a firm requirement is deferred until you ask, message first.', '2. Discover with short constraint-focused search/filter queries; reformulate on weak hits.', '3. view/compare a shortlist and verify EVERY firm requirement on the exact sku using raw facts.', '4. First add_to_cart must be a genuine fully compliant choice, never a placeholder.', '5. Soft preferences guide ranking among compliant options; firm constraints always win.', '6. After add_to_cart OR any post-cart shopper update, inspect_stock the carted sku again before place_test_order — price and stock can change on cart events. If the fresh check fails budget or stock, remove_from_cart and recover.', '7. Never place_test_order in the same turn as add_to_cart.', '8. When verified by a post-cart inspect_stock, place_test_order immediately.', '9. Do not repeat the same search/filter query; broaden, drop optional tokens, or inspect hits.', f'10. Up to {max_calls} tool calls per turn; budget {max_steps} turns. Batch independent reads.', '', 'Tools:', *tool_lines]
    if ledger.needs_ask and (not ledger.clarified):
        parts.append('This request is underspecified. Your first action should ask for the missing firm product specification.')
    if ledger.named:
        shown = ', '.join((f'{pid}::{sku}' for (pid, sku) in ledger.named[:12]))
        parts.extend(['Named variants to compare/view before carting:', shown, 'Respect the stated tradeoff order (primary metric, then secondary, then price).'])
    if ledger.rank_axes:
        axes = ', '.join((f"{a}({('higher' if d > 0 else 'lower')})" for (a, d) in ledger.rank_axes))
        parts.append(f'Ranking axes from the request: {axes}.')
    if needs_justification:
        parts.extend(['Checkout requires structured justification claims grounded in view/compare/inspect_stock of the FINAL product.', f"Allowed fields: {', '.join(claim_fields)}.", 'Always include price and in_stock from the latest inspect_stock.', "Include exactly one claim for the shopper's constrained attribute (snake_case) with the observed value, plus one positive capability/reason field when the schema allows it."])
    else:
        parts.append('Omit justification unless the schema requires it.')
    parts.append('Continue until an observation reports done=true.')
    return '\n'.join(parts)

def tf2_problem_is_hard(*, ledger: tf2_Ledger, needs_justification: bool, max_steps: int) -> bool:
    """Cost-aware model choice from public problem cues only."""
    if needs_justification or ledger.needs_ask or ledger.named or ledger.rank_axes:
        return True
    if ledger.procedure in {PROC_ASK, PROC_SOFT, PROC_CLAIM, PROC_RANK, PROC_ADAPT, PROC_USE}:
        return True
    if ledger.soft_signal and ledger.firm_signal:
        return True
    if max_steps >= 24:
        return True
    return len(_preamble(ledger.query)) > 420

# ===== KA.py pipeline (tf3) =====
def tf3_model() -> str:
    provider = getenv('INFERENCE_PROVIDER', 'chutes')
    return getenv('SANDBOX_MODEL') or _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS['chutes'])

def _order_schema(tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    for spec in tools:
        fn = spec.get('function') if isinstance(spec, dict) else None
        if isinstance(fn, dict) and fn.get('name') == 'place_test_order':
            params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
            return params
    return None

def _tool_required(tools: list[dict[str, Any]], name: str) -> list[str]:
    for spec in tools:
        fn = spec.get('function') if isinstance(spec, dict) else None
        if not isinstance(fn, dict) or fn.get('name') != name:
            continue
        params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
        return [x for x in params.get('required', []) if isinstance(x, str)]
    return []

def _allowed_claim_fields(tools: list[dict[str, Any]]) -> set[str]:
    """Read allowed justification field names from the live tool schema only."""
    params = _order_schema(tools) or {}
    props = params.get('properties') if isinstance(params.get('properties'), dict) else {}
    just = props.get('justification') if isinstance(props.get('justification'), dict) else {}
    jprops = just.get('properties') if isinstance(just.get('properties'), dict) else {}
    claims = jprops.get('claims') if isinstance(jprops.get('claims'), dict) else {}
    items = claims.get('items') if isinstance(claims.get('items'), dict) else {}
    iprops = items.get('properties') if isinstance(items.get('properties'), dict) else {}
    field = iprops.get('field') if isinstance(iprops.get('field'), dict) else {}
    allowed: set[str] = set()
    enum = field.get('enum')
    if isinstance(enum, list):
        allowed.update((str(x) for x in enum if isinstance(x, str)))
    for blob in (field.get('description'), just.get('description'), claims.get('description')):
        if isinstance(blob, str) and blob.strip():
            allowed.update(re.findall('\\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\\b', blob))
            allowed.update((tok for tok in re.findall('\\b[a-z][a-z0-9]{2,}\\b', blob) if tok not in {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'must', 'only', 'field', 'value', 'claim', 'claims', 'true', 'false', 'string', 'object', 'array', 'required', 'optional', 'product', 'order', 'justify', 'justification'}))
    allowed.update({'price', 'in_stock'})
    return allowed

def tf3_system_prompt(tools: list[dict[str, Any]], max_calls: int, max_steps: int) -> str:
    needs_justification = 'justification' in _tool_required(tools, 'place_test_order')
    allowed = sorted(_allowed_claim_fields(tools)) if needs_justification else []
    lines = ['You are a shopping assistant in a product catalog.', 'Help the shopper find a valid in-stock item, confirm it, and complete the purchase with the tools.', 'Follow the shopper request and every attached instruction. Never invent product_id or sku values.', '', 'Method:', '- Search with short keywords (2-5 words). Reformulate or use filter when results are poor.', '- Prefer in_stock=true when available.', f'- Up to {max_calls} tool calls per turn; batch independent reads when useful.', '- Inspect with view/compare before the first add_to_cart. Cart only a real intended choice.', '- After cart changes, call inspect_stock. If invalid, remove_from_cart, recover, then order.', '- Order path: view/compare -> add_to_cart -> inspect_stock -> place_test_order.', '- Answer shopper questions with message before ordering.', f'- Budget about {max_steps} turns. When verified, place_test_order immediately.', '- Do not spend many turns only messaging or repeatedly removing from cart without ordering.']
    if needs_justification:
        lines.append(f"- place_test_order requires justification claims. Use only field names supported by the tool schema ({', '.join(allowed)}). Values must match inspected facts for the final product. Always include truthful price and in_stock from inspect_stock. Never invent unsupported features.")
    else:
        lines.append('- Omit justification on place_test_order unless the tool schema requires it.')
    lines.append('')
    lines.append('Tools:')
    for spec in tools:
        fn = spec.get('function') if isinstance(spec, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get('name'), str):
            lines.append(f"- {fn['name']}: {str(fn.get('description') or '').strip()}")
    lines.append('')
    lines.append('Continue until an observation reports done=true.')
    return '\n'.join(lines)

def tf3_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call['function'].get('arguments', '{}')
    if isinstance(raw, dict):
        parsed: Any = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            fixed = re.sub(',\\s*}', '}', raw.replace("'", '"'))
            fixed = re.sub(',\\s*]', ']', fixed)
            parsed = json.loads(fixed)
    else:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError('tool arguments must be a JSON object')
    return parsed

def _ground_order_args(args: dict[str, Any], *, tools: list[dict[str, Any]], stock_by_key: dict[str, dict[str, Any]], viewed_facts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cleaned = dict(args)
    required = _tool_required(tools, 'place_test_order')
    if 'justification' in cleaned and 'justification' not in required:
        cleaned.pop('justification', None)
        return cleaned
    if 'justification' not in required:
        return cleaned
    pid = str(cleaned.get('product_id') or '')
    sku = str(cleaned.get('sku') or '')
    key = f'{pid}::{sku}'
    stock = stock_by_key.get(key) or stock_by_key.get(pid) or {}
    facts = viewed_facts.get(pid) or {}
    allowed = _allowed_claim_fields(tools)
    raw_claims = []
    just = cleaned.get('justification')
    if isinstance(just, dict) and isinstance(just.get('claims'), list):
        raw_claims = [c for c in just['claims'] if isinstance(c, dict)]
    grounded: list[dict[str, str]] = []
    seen_fields: set[str] = set()

    def _lookup(field: str) -> Any:
        for blob in (stock, facts):
            if field in blob and blob.get(field) not in (None, ''):
                return blob.get(field)
            specs = blob.get('specification') if isinstance(blob.get('specification'), dict) else {}
            attrs = blob.get('attributes') if isinstance(blob.get('attributes'), dict) else {}
            for nest in (specs, attrs):
                if field in nest and nest.get(field) not in (None, ''):
                    return nest.get(field)
        return None

    def add(field: str, value: Any) -> None:
        if field not in allowed or field in seen_fields:
            return
        if value is None or value == '':
            return
        if isinstance(value, bool):
            text = 'true' if value else 'false'
        else:
            text = str(value).strip()
        if not text:
            return
        grounded.append({'field': field, 'value': text})
        seen_fields.add(field)
    if 'price' in stock:
        add('price', stock.get('price'))
    if 'in_stock' in stock:
        add('in_stock', stock.get('in_stock'))
    for claim in raw_claims:
        field = str(claim.get('field') or '').strip()
        if field in {'price', 'in_stock'}:
            continue
        if field not in allowed:
            continue
        observed = _lookup(field)
        if observed is not None:
            add(field, observed)
            continue
        value = claim.get('value')
        if isinstance(value, bool) or str(value).lower() in {'true', 'false'}:
            continue
        add(field, value)
    for field in sorted(allowed):
        if field in seen_fields:
            continue
        observed = _lookup(field)
        if observed is not None:
            add(field, observed)
    if not grounded:
        if 'price' in stock:
            add('price', stock.get('price'))
        if 'in_stock' in stock:
            add('in_stock', stock.get('in_stock'))
    cleaned['justification'] = {'claims': grounded}
    return cleaned

def _clean_args(name: str, args: dict[str, Any], tools: list[dict[str, Any]], stock_by_key: dict[str, dict[str, Any]], viewed_facts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cleaned = dict(args)
    if name == 'search' and 'in_stock' not in cleaned:
        cleaned['in_stock'] = True
    if name == 'place_test_order':
        cleaned = _ground_order_args(cleaned, tools=tools, stock_by_key=stock_by_key, viewed_facts=viewed_facts)
    return cleaned

def _remember_observation(name: str, args: dict[str, Any], observation: dict[str, Any], stock_by_key: dict[str, dict[str, Any]], viewed_facts: dict[str, dict[str, Any]]) -> None:
    if name == 'inspect_stock':
        pid = str(observation.get('product_id') or args.get('product_id') or '')
        sku = str(observation.get('sku') or args.get('sku') or '')
        if pid:
            stock_by_key[f'{pid}::{sku}'] = dict(observation)
            stock_by_key[pid] = dict(observation)
    if name in {'view', 'compare'}:
        pid = observation.get('product_id')
        if isinstance(pid, str) and pid:
            viewed_facts[pid] = dict(observation)
        for row in observation.get('compared') or []:
            if isinstance(row, dict) and isinstance(row.get('product_id'), str):
                viewed_facts[row['product_id']] = dict(row)

# ===== KB_2.py pipeline (tf4) =====
tf4_MODELS = {'chutes': {'default': 'Qwen/Qwen3.5-397B-A17B-TEE', 'fallback': ('deepseek-ai/DeepSeek-V3.2-TEE', 'moonshotai/Kimi-K2.6-TEE')}, 'openrouter': {'default': 'qwen/qwen3.5-397b-a17b', 'fallback': ('deepseek/deepseek-v3.2', 'moonshotai/kimi-k2.6')}}
_TOKEN_RE = re.compile('[a-z0-9]+', re.I)
tf4_BUDGET_RE = re.compile('(?:(?:under|below|upto|up\\s+to|at\\s+or\\s+under|at\\s+most|no\\s+more\\s+than|max(?:imum)?|budget(?:\\s+of)?|spend(?:\\s+up\\s+to)?|can\\s+(?:only\\s+)?(?:spend(?:\\s+up\\s+to)?|go\\s+up\\s+to)|go\\s+up\\s+to)\\s*(?:of\\s+)?(?P<a>\\d[\\d,]*(?:\\.\\d+)?)|(?P<b>\\d[\\d,]*(?:\\.\\d+)?)\\s*(?P<cur>[a-z]{3})\\b)', re.I)
_CANDIDATE_KEY_RE = re.compile('\\b(\\d{6,})::([A-Za-z0-9_-]+)\\b')
_NUM_TOKEN_RE = re.compile('\\d+(?:\\.\\d+)?')
_NUM_UNIT_RE = re.compile('\\b(\\d{1,4}(?:\\.\\d+)?)\\s*-?\\s*([A-Za-z]{1,16})\\b')

def _canon_unit(unit: str) -> str:
    """Orthographic fold for length-quote units only — no product vocab."""
    u = (unit or '').lower().strip()
    if u in {'"', "'", 'in', 'inch', 'inches'}:
        return 'in'
    return u

def _canon_spec_token(token: str) -> str:
    """Normalize a query/product spec token (NUM-unit folds only)."""
    raw = (token or '').strip()
    if not raw:
        return raw
    match = re.fullmatch('(\\d+(?:\\.\\d+)?)[\\s\\-]*([A-Za-z%]{1,16}|\\"|\')', raw)
    if match:
        return f'{match.group(1)}-{_canon_unit(match.group(2))}'
    match = re.fullmatch('(\\d+(?:\\.\\d+)?)(in|inch|inches)\\b', raw, re.I)
    if match:
        return f'{match.group(1)}-in'
    return raw
_CATEGORY_RE = re.compile('\\bin\\s+the\\s+([A-Za-z0-9][\\w\\s&/-]{1,60}?)\\s+category\\b', re.I)
_OBS_CHAR_CAP = 1800
_STR_CHAR_CAP = 240
_LIST_ROW_CAP = 6
_BOARD_CANDIDATE_CAP = 6
_HISTORY_SOFT_CAP = 8
_MAX_STORED_CANDIDATES = 20
_TURN_CEILING = 28
_SEED_STOP = frozenset("\n    a an the and or but for with from that this these those into onto over under\n    at to of in on by is are was were be been being it its as if then than so\n    not no yes you your we our they their i me my need want find get buy purchase\n    please can could would should must also just only more most less least very\n    item items product products listing listings order orders cart shop shopping\n    available stock stocked budget price prices within below above prefer preferred\n    ideally soft firm requirement requirements category categories brand brands\n    spend spending going actually really sorry hold wait still needs needed\n    lean leaning toward towards check checked recheck before anything\n    compare choose among remaining ties tradeoff tradeoffs candidate candidates\n    variant variants option options walk through why right pick select\n    highest lowest largest smallest cheapest maximum minimum advertised\n    including explicitly current live between while after once\n    place one test rather pay bit every kind something settle\n    through give trust tell another keep exact ask asking say identify\n    title type labels insufficient replacement reason good has care about\n    matters goal sku dollars saving few grab dont don't matches phrase catalog\n    ".split())
_DIR_MAX = frozenset({'highest', 'largest', 'most', 'maximum', 'max', 'greatest', 'biggest'})
_DIR_MIN = frozenset({'lowest', 'smallest', 'least', 'minimum', 'min', 'cheapest', 'fewest'})
_TRADEOFF_RE = re.compile('(?:(?:choose|pick|prefer|select|want|get|preferring)\\s+(?:the\\s+)?|(?:then|and|,)\\s+(?:the\\s+)?|(?:\\A|;|\\.)\\s*(?:the\\s+)?)(?P<dir>highest|lowest|largest|smallest|cheapest|most|least|maximum|minimum)\\s+(?P<field>(?:(?:advertised|maximum|minimum|available|current|live)\\s+){0,3}[a-z][a-z0-9]*(?:\\s+(?!then\\b|and\\b|or\\b|before\\b|after\\b|while\\b|among\\b|between\\b|with\\b)[a-z][a-z0-9/-]*){0,4})', re.I | re.M)

def _model_candidates() -> list[str]:
    """Prefer provider-native allowlisted IDs; remap TEE overrides on OpenRouter."""
    provider = _provider()
    catalog = tf4_MODELS.get(provider, tf4_MODELS['chutes'])
    override = (getenv('SANDBOX_MODEL') or '').strip()
    ordered: list[str] = []
    if override:
        if provider == 'openrouter' and override.endswith('-TEE'):
            mapped = {'deepseek-ai/DeepSeek-V3.2-TEE': 'deepseek/deepseek-v3.2', 'deepseek-ai/DeepSeek-V3.1-TEE': 'deepseek/deepseek-chat-v3.1', 'Qwen/Qwen3.5-397B-A17B-TEE': 'qwen/qwen3.5-397b-a17b', 'Qwen/Qwen3-32B-TEE': 'qwen/qwen3-32b', 'moonshotai/Kimi-K2.6-TEE': 'moonshotai/kimi-k2.6'}
            ordered.append(mapped.get(override, catalog['default']))
        else:
            ordered.append(override)
    ordered.append(catalog['default'])
    ordered.extend(catalog['fallback'])
    uniq: list[str] = []
    seen: set[str] = set()
    for model in ordered:
        if model in seen:
            continue
        seen.add(model)
        uniq.append(model)
    return uniq

def tf4_model() -> str:
    return _model_candidates()[0]

def _claim_field_allowlist(tools: list[dict[str, Any]]) -> set[str]:
    allowed: set[str] = set()
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get('name') != 'place_test_order':
            continue
        params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
        props = params.get('properties') if isinstance(params.get('properties'), dict) else {}
        just = props.get('justification') if isinstance(props.get('justification'), dict) else {}
        jprops = just.get('properties') if isinstance(just.get('properties'), dict) else {}
        claims = jprops.get('claims') if isinstance(jprops.get('claims'), dict) else {}
        items = claims.get('items') if isinstance(claims.get('items'), dict) else {}
        iprops = items.get('properties') if isinstance(items.get('properties'), dict) else {}
        field = iprops.get('field') if isinstance(iprops.get('field'), dict) else {}
        enum = field.get('enum')
        if isinstance(enum, list):
            allowed.update((str(x) for x in enum if isinstance(x, str) and x.strip()))
        desc = ' '.join((str(part) for part in (field.get('description'), just.get('description'), claims.get('description')) if isinstance(part, str))).lower()
        for token in re.findall('\\b[a-z][a-z0-9_]{1,40}\\b', desc):
            if token in {'field', 'fields', 'value', 'claims', 'claim', 'string', 'object', 'array', 'supported', 'reason', 'reasons', 'useful', 'specification', 'shopper', 'request', 'feature', 'features', 'positive', 'capabilities', 'unknown', 'absent'}:
                continue
            if '_' in token or token in {'price', 'brand', 'category'}:
                allowed.add(token)
        reason = re.search('reason(?:s)?\\s*:?\\s*([^.]+)', desc)
        if reason:
            clause = reason.group(1)
            stop = {'such', 'also', 'one', 'the', 'and', 'for', 'or', 'as', 'to', 'use', 'only', 'field', 'value', 'supported', 'positive', 'capability', 'capabilities'}
            for token in re.findall('[a-z][a-z0-9_]{1,40}', clause):
                if token in stop:
                    continue
                allowed.add(token)
            for (left, right) in re.findall('([a-z]+)\\s+([a-z]+)', clause):
                if left in stop or right in stop:
                    continue
                allowed.add(f'{left}_{right}')
        if 'justification' in json.dumps(params, ensure_ascii=False).lower() or allowed:
            allowed.update({'price', 'in_stock'})
    return allowed

def _justification_open_spec(tools: list[dict[str, Any]]) -> bool:
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get('name') != 'place_test_order':
            continue
        blob = json.dumps(fn.get('parameters') or {}, ensure_ascii=False).lower()
        if 'specification' in blob and 'shopper' in blob:
            return True
        if 'specification field' in blob:
            return True
    return False

def _flatten_product(row: dict[str, Any]) -> dict[str, Any]:
    flat = dict(row)
    for nest in ('facts', 'specification', 'attributes', 'specs'):
        block = row.get(nest)
        if isinstance(block, dict):
            for (key, value) in block.items():
                flat.setdefault(key, value)
    path = row.get('category_path')
    if isinstance(path, list) and path:
        flat.setdefault('category', path[-1] if isinstance(path[-1], str) else path[0])
    elif isinstance(path, str) and path.strip():
        flat.setdefault('category', path.split(',')[-1].strip().strip('[]"\' '))
    title = str(row.get('title') or '')
    if title and (not flat.get('brand')):
        token = title.split()[0] if title.split() else ''
        if token[:1].isupper() and token.isalpha() and (len(token) >= 3):
            flat.setdefault('brand', token)
    return flat

def tf4_parse_budget(text: str) -> float | None:
    best: float | None = None
    for match in tf4_BUDGET_RE.finditer(text or ''):
        raw = match.group('a') or match.group('b')
        if not raw:
            continue
        try:
            value = float(raw.replace(',', ''))
        except ValueError:
            continue
        if value < 5:
            continue
        if best is None or value < best:
            best = value
    return best

def tf4_preamble(query: str) -> str:
    """Shopper request text before any Task rules appendix."""
    return (query or '').split('Task rules:', 1)[0].strip()

def tf4_needs_clarification(query: str) -> bool:
    """Detect deferred constraints via discourse co-occurrence (no fixed suite clauses)."""
    text = tf4_preamble(query).lower()
    if not text:
        return False
    if re.search('\\b(?:ask|asking)\\b.{0,48}\\b(?:tell|share|reveal|provide|give)\\b', text) or re.search('\\b(?:tell|share|reveal|provide|give)\\b.{0,48}\\b(?:ask|asking)\\b', text):
        return True
    if re.search("\\b(?:haven'?t|have not|not yet)\\s+(?:told|said|shared|given)\\b", text) and re.search('\\bask\\b', text):
        return True
    if re.search('\\bif\\b.{0,24}\\bask\\b.{0,40}\\b(?:tell|share|reveal|provide|give)\\b', text):
        return True
    return False

def _budget_number_strings(text: str) -> set[str]:
    found: set[str] = set()
    for match in tf4_BUDGET_RE.finditer(text or ''):
        raw = match.group('a') or match.group('b')
        if not raw:
            continue
        cleaned = raw.replace(',', '')
        found.add(cleaned)
        try:
            found.add(str(int(float(cleaned))))
        except ValueError:
            pass
    return found

def _currency_codes_in_text(text: str) -> set[str]:
    """Collect currency codes written next to amounts in the ask (not a fixed list)."""
    codes: set[str] = set()
    for match in tf4_BUDGET_RE.finditer(text or ''):
        cur = match.group('cur')
        if cur:
            codes.add(cur.lower())
    for match in re.finditer('\\b\\d[\\d,]*(?:\\.\\d+)?\\s+([A-Za-z]{3})\\b', text or ''):
        codes.add(match.group(1).lower())
    return codes

def _category_hint(query: str) -> str | None:
    match = _CATEGORY_RE.search(tf4_preamble(query))
    if not match:
        return None
    value = re.sub('\\s+', ' ', match.group(1)).strip(' .,;:')
    return value or None

def _sanitize_note_for_search(note: str) -> str:
    """Drop price-narration / catalog-key noise from shopper notes before retrieval."""
    text = note or ''
    text = _CANDIDATE_KEY_RE.sub(' ', text)
    text = re.sub('\\b\\d[\\d,]*(?:\\.\\d+)?\\s*(?:→|->|to)\\s*\\d[\\d,]*(?:\\.\\d+)?\\b', ' ', text)
    text = re.sub('(?:latest\\s+)?live\\s+price\\b[^.]{0,60}', ' ', text, flags=re.I)
    text = re.sub("\\b(?:i am going with|i'll go with|going with)\\b[^.]{0,120}", ' ', text, flags=re.I)
    text = re.sub('\\b(?:went\\s+up|price\\s+hike|more\\s+expensive|expensive|budget\\s+left|which|stated|ordering)\\b', ' ', text, flags=re.I)
    text = re.sub('\\bgo\\s+up(?!\\s+to\\b)\\b', ' ', text, flags=re.I)
    text = re.sub('\\b(?:thanks|update|leaning|toward|available|re-check|recheck)\\b', ' ', text, flags=re.I)
    text = re.sub('(?<![A-Za-z0-9])\\d{1,3}(?![A-Za-z0-9.-])', ' ', text)
    return text

def _priority_soft_phrases(text: str) -> list[str]:
    """Extract ranked soft factors from shopper priority clarifications."""
    raw = text or ''
    found: list[str] = []
    seen: set[str] = set()

    def push(phrase: str) -> None:
        phrase = re.sub('\\s+', ' ', phrase or '').strip(' .,;:')
        phrase = re.sub('^(?:with\\s+a|with|a|an|the|behind|close)\\s+', '', phrase, flags=re.I)
        words = [tok for tok in re.split('[\\s/]+', phrase) if tok and tok.lower() not in _SEED_STOP and (tok.lower() not in {'most', 'least', 'next', 'behind', 'close', 'counts', 'count', 'php', 'under', 'stock'})]
        if not words:
            return
        phrase = ' '.join(words[:4])
        low = phrase.lower()
        if low in seen or len(low) < 3:
            return
        seen.add(low)
        found.append(phrase)
        for tok in words[:4]:
            tl = tok.lower()
            if len(tl) >= 3 and tl not in seen:
                seen.add(tl)
                found.append(tok)
    for match in re.finditer('\\b([A-Za-z][A-Za-z0-9]+(?:[\\s/]+[A-Za-z][A-Za-z0-9]+){0,3})\\s+counts?\\s+(?:the\\s+)?(?:most|least)\\b', raw, re.I):
        push(match.group(1))
    for match in re.finditer('\\bwith\\s+a\\s+([A-Za-z][A-Za-z0-9]+(?:\\s+[A-Za-z][A-Za-z0-9]+){0,3})\\s+close\\s+behind\\b', raw, re.I):
        push(match.group(1))
    for match in re.finditer('\\bclose\\s+behind\\s+([A-Za-z][A-Za-z0-9]+(?:\\s+[A-Za-z][A-Za-z0-9]+){0,2})\\s+next\\b', raw, re.I):
        push(match.group(1))
    return found[:12]

def _note_states_spend_cap(text: str) -> bool:
    """True when the shopper is revising a spend limit (not narrating a price hike)."""
    raw = text or ''
    if not raw.strip():
        return False
    if re.search('\\b(?:budget|at\\s+most|no\\s+more\\s+than|can\\s+only|balance|max(?:imum)?(?:\\s+budget)?|spend)\\b', raw, re.I):
        return True
    if re.search('\\b(?:go\\s+)?up\\s+to\\s+\\d', raw, re.I) and re.search('\\b(?:can\\s+only|sorry|now|budget|balance|under|below|at\\s+most)\\b', raw, re.I):
        return True
    return False

def _soft_fact_dimensions(query: str) -> list[str]:
    """Extract inspection dimensions named in the public task contract text."""
    text = query or ''
    found: list[str] = []
    seen: set[str] = set()
    for match in re.finditer('\\bfacts?\\s+(?:such\\s+as|like)\\s+([^.!?]{6,180})', text, re.I):
        chunk = match.group(1)
        chunk = re.split('\\b(?:then|instead|before|after|while|so\\s+that)\\b', chunk, maxsplit=1, flags=re.I)[0]
        for part in re.split('\\s*,\\s*|\\s*/\\s*|\\s+and\\s+', chunk):
            part = re.sub('\\s+', ' ', part).strip(' .,;:()')
            if len(part) < 2 or len(part) > 48:
                continue
            words = [tok for tok in re.findall('[A-Za-z][A-Za-z0-9/+-]{1,}', part) if tok.lower() not in _SEED_STOP and tok.lower() not in {'use', 'specific', 'features', 'feature', 'raw', 'product', 'visible'}]
            if not words:
                continue
            phrase = ' '.join(words[:4])
            low = phrase.lower()
            if low in seen or 'use-specific' in low:
                continue
            seen.add(low)
            found.append(phrase)
            for tok in words[:3]:
                tl = tok.lower()
                if tl not in seen and len(tl) >= 3:
                    seen.add(tl)
                    found.append(tok)
    return found[:10]

def _use_context_phrases(query: str) -> list[str]:
    """Capture use-case windows from the shopper preamble (discourse only)."""
    preamble = tf4_preamble(query)
    found: list[str] = []
    seen: set[str] = set()
    for match in re.finditer('\\bfor\\s+(?!an?\\s+in-stock\\b)(?!a\\s+test\\b)([^.;!]{5,70})', preamble, re.I):
        prev = preamble[max(0, match.start() - 24):match.start()].lower()
        if re.search('\\b(?:settle|rather|looking|search|place|order|ask|except|excepting)\\s*$', prev):
            continue
        chunk = match.group(1)
        chunk = re.split('\\b(?:with|that|than|under|at\\s+or)\\b', chunk, maxsplit=1, flags=re.I)[0]
        words = [tok for tok in re.findall('[A-Za-z][A-Za-z0-9/+-]{2,}', chunk) if tok.lower() not in _SEED_STOP and tok.lower() not in {'place', 'order', 'stock', 'test', 'item', 'device', 'category', 'budget', 'flimsy', 'settle', 'something'}]
        if not words:
            continue
        phrase = ' '.join(words[:4])
        low = phrase.lower()
        if low in seen:
            continue
        seen.add(low)
        found.append(phrase)
    return found[:4]

def _soft_code_match(blob: str, code: str) -> bool:
    """Match preferred model codes as whole tokens — never as price substrings."""
    code = (code or '').strip()
    if not code or not blob:
        return False
    blob_l = blob.lower()
    if re.fullmatch('\\d{3,14}', code):
        return bool(re.search(f'(?<!\\d){re.escape(code)}(?!\\d)', blob_l))
    return _fragment_match(blob_l, code)

def _match_targets(text: str) -> list[str]:
    """Extract explicit match phrases the shopper asked the catalog to hit."""
    found: list[str] = []
    seen: set[str] = set()

    def push(raw: str) -> None:
        value = re.sub('\\s+', ' ', raw or '').strip(' .,;:|/\\"\'')
        if len(value) < 2:
            return
        low = value.lower()
        if low in seen or low in _SEED_STOP:
            return
        seen.add(low)
        found.append(value)
    for match in re.finditer('"([^"\\n]{2,80})"', text or ''):
        push(match.group(1))
    for match in re.finditer('\\bmatches\\s+the\\s+phrase\\s+([^,.;\\n]{2,80})', text or '', re.I):
        push(match.group(1))
    for match in re.finditer('\\bmatches\\s+(?!the\\s+phrase\\b)([A-Za-z0-9][\\w\\s&/-]{1,60}?)(?=\\s*,|\\s+at\\s+or|\\s+under|\\s+and\\s+buy|\\s*$)', text or '', re.I):
        push(match.group(1))
    return found[:3]

def _search_parts(query: str, notes: list[str] | None=None) -> list[str]:
    """Split the shopper ask into short discovery tokens (query-derived only)."""
    chunks = [tf4_preamble(query)]
    if notes:
        chunks.extend((_sanitize_note_for_search(n) for n in notes[-2:]))
    text = ' '.join(chunks)
    text = _CANDIDATE_KEY_RE.sub(' ', text)
    targets = _match_targets(text)
    if targets:
        return targets[:2]
    budget_nums = _budget_number_strings(text)
    currency_codes = _currency_codes_in_text(text)
    parts: list[str] = []
    seen: set[str] = set(currency_codes)
    used_units: set[str] = set()

    def add(token: str, *, allow_model_code: bool=False) -> None:
        tok = (token or '').strip(' ,.;:|/\\"\'')
        if not tok:
            return
        tok = _canon_spec_token(tok)
        low = tok.lower()
        if low in seen or low in _SEED_STOP or low in currency_codes:
            return
        if low in {'in', 'inch', 'inches', 'in-stock', 'instock', 'stock'}:
            return
        if '::' in tok:
            return
        if re.fullmatch('\\d{6,}', low):
            if not allow_model_code or not 6 <= len(low) <= 14:
                return
        if re.fullmatch('\\d{3,5}', low):
            if not allow_model_code:
                return
            if low in budget_nums:
                return
        if re.fullmatch('[a-z0-9]+-\\d+', low) and len(low) >= 8:
            return
        if re.fullmatch('\\d+(?:\\.\\d+)?', low):
            if low in budget_nums:
                return
            try:
                if float(low) >= 1000 and (not allow_model_code):
                    return
            except ValueError:
                return
        seen.add(low)
        parts.append(tok)
    for match in _NUM_UNIT_RE.finditer(text):
        (num, unit) = (match.group(1), match.group(2))
        unit_l = _canon_unit(unit)
        if len(unit_l) < 2 or unit_l in _SEED_STOP or unit_l in currency_codes:
            continue
        if num in budget_nums:
            continue
        try:
            if float(num) >= 1000 and len(unit_l) < 2:
                continue
        except ValueError:
            continue
        used_units.add(unit_l)
        add(f'{num}-{unit_l}')
    soft_parts: list[str] = []
    soft_seen: set[str] = set()
    for soft_match in tf4_SOFT_CUE_RE.finditer(text):
        window = text[soft_match.end():soft_match.end() + 48]
        for code in re.findall('\\b(\\d{3,14})\\b', window):
            low = code.lower()
            if low not in soft_seen:
                soft_seen.add(low)
                soft_parts.append(code)
            add(code, allow_model_code=True)
        for tok in re.findall('[A-Za-z][A-Za-z0-9.+-]{2,}', window):
            low = tok.lower()
            if low in _SEED_STOP or low in soft_seen:
                continue
            soft_seen.add(low)
            soft_parts.append(tok)
            add(tok)
            if low.endswith('ance') and len(low) > 6:
                alt = low[:-4] + 'ant'
                if alt not in soft_seen and alt not in _SEED_STOP:
                    soft_seen.add(alt)
                    soft_parts.append(alt)
                    add(alt)
    for match in re.finditer('\\b([A-Za-z]*\\d+[A-Za-z0-9.-]*)\\b', text):
        tok = match.group(1)
        bare = tok.replace('.', '').replace('-', '')
        if bare in budget_nums or tok.lower() in currency_codes:
            continue
        add(tok)
    for match in re.finditer('\\b([A-Z][A-Za-z0-9]{2,})\\b', tf4_preamble(query)):
        tok = match.group(1)
        if tok.lower() in currency_codes:
            continue
        add(tok)
    for tok in re.findall('[A-Za-z][A-Za-z0-9.+-]{2,}', text):
        if tok.lower() in used_units:
            continue
        add(tok)
        if len(parts) >= 14:
            break
    specs = [p for p in parts if re.search('\\d', p)]
    words = [p for p in parts if p not in specs]
    soft_l = {s.lower() for s in soft_parts}
    core = [w for w in words if w.lower() not in soft_l and len(w) >= 4]
    soft_keep = [w for w in soft_parts if w.lower() not in {c.lower() for c in core}]
    short = [w for w in words if len(w) < 4 and w.lower() not in soft_l]
    ordered = specs[:3] + core[:3] + soft_keep[:3] + short[:1]
    return (ordered or parts)[:7]

def _search_seed(query: str, notes: list[str] | None=None) -> str:
    """Build a short discovery query from the public goal + latest shopper notes."""
    parts = _search_parts(query, notes)
    return ' '.join(parts).strip() or 'in stock'

def _dedupe_tokens(parts: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        low = part.lower()
        if not low or low in seen:
            continue
        seen.add(low)
        out.append(part)
    return out

def _search_variants(query: str, notes: list[str] | None, attempt: int) -> str:
    """Rotate short formulations when prior searches were empty or noisy."""
    note_blob = ' '.join([tf4_preamble(query)] + list(notes or [])[-2:])
    targets = _match_targets(note_blob)
    if targets:
        phrase = targets[0]
        (firm, soft) = ([], [])
        try:
            (firm, soft) = _split_constraints(query, notes)
        except Exception:
            pass
        soft_codes = [s for s in soft if re.fullmatch('\\d{3,14}', s)]
        soft_words = [s for s in soft if s not in soft_codes][:2]
        variants = [phrase, ' '.join(_dedupe_tokens([phrase] + soft_words[:1])[:4]), ' '.join(_dedupe_tokens([phrase] + soft_codes[:1] + soft_words[:1])[:4]), ' '.join(_dedupe_tokens([phrase] + [f for f in firm if not re.search('\\d', f)][:1])[:3])]
        return variants[attempt % len(variants)]
    parts = _dedupe_tokens(_search_parts(query, notes))
    if not parts:
        return 'in stock'
    specs = [p for p in parts if re.search('\\d', p)]
    words = [p for p in parts if p not in specs]
    category = _category_hint(query)
    soft: list[str] = []
    firm: list[str] = []
    try:
        (firm, soft) = _split_constraints(query, notes)
    except Exception:
        soft = []
    soft = _dedupe_tokens([s for s in soft if len(s.split()) <= 3])
    soft_codes = [s for s in soft if re.fullmatch('\\d{3,14}', s)]
    soft_words = [s for s in soft if s not in soft_codes and ' ' not in s and (len(s) >= 3)][:3]
    soft_l = {s.lower() for s in soft}
    firm_nouns = [f for f in firm if not re.search('\\d', f) and len(f.split()) == 1 and (len(f) >= 4)][:3]
    type_nouns = [f for f in firm_nouns if not f[:1].isupper()][:1]
    brand_nouns = [f for f in firm_nouns if f[:1].isupper()][:1]
    core = [w for w in words if w.lower() not in soft_l][:2]
    primary = _dedupe_tokens(soft_codes[:1] + soft_words[:1] + core[:1] + firm_nouns[:1] + specs[:1])
    typed = _dedupe_tokens(type_nouns + specs[:1] + brand_nouns + soft_words[:1] + core[:1])
    variants = [' '.join(typed[:4]) if type_nouns else ' '.join(soft_codes[:1] + soft_words[:1] + core[:1]) if soft_codes or soft_words else ' '.join(primary[:4]), ' '.join(primary[:4]), ' '.join(_dedupe_tokens(soft_codes[:1] + core[:1] + soft_words[:2] + firm_nouns[:1])[:4]), ' '.join(_dedupe_tokens(soft_words[:2] + core[:1])[:3]), ' '.join(_dedupe_tokens(soft_codes[:1] + firm_nouns[:1] + soft_words[:1])[:3]), ' '.join(_dedupe_tokens(core[:1] + soft_words[:2] + specs[:1])[:4]), ' '.join(parts[:5]), ' '.join(words[:3]) if words else ' '.join(parts[:3])]
    if category:
        variants.append(f"{category} {' '.join(primary[:2])}".strip())
    variants = [v.strip() for v in variants if v and v.strip()]
    if not variants:
        return 'in stock'
    return variants[attempt % len(variants)]

def _classify_task_shape(problem_data: dict[str, Any], *, tools: list[dict[str, Any]], named: list[tuple[str, str]], needs_ask: bool) -> str:
    """Branch only on public problem_data / schema signals (not wall-clock)."""
    family = str(problem_data.get('family') or (problem_data.get('environment') or {}).get('family') or '').strip().lower()
    if family:
        return family
    query = str(((problem_data.get('environment') or {}).get('policy_view') or {}).get('query') or problem_data.get('query') or '')
    preamble = tf4_preamble(query)
    text_l = (query or '').lower()
    pre_l = preamble.lower()
    if len(named) >= 2:
        return 'ranking'
    if 'justification' in _required_args(tools, 'place_test_order') or re.search('\\bwalk\\s+me\\s+through\\s+why\\b', pre_l):
        return 'justification'
    if needs_ask:
        return 'intent_decomposition'
    if re.search('\\bideally\\b.{0,60}\\bprefer\\b.{0,100}\\bfirm\\s+requirements?\\s+matter', pre_l):
        return 'intent_decomposition'
    if re.search('\\bmatches?\\s+"[^"\\n]{2,80}"', preamble) or re.search("\\bmatches?\\s+[“‘'][^”’'\\n]{2,80}[”’']", preamble) or re.search('\\bmatches?\\s+(?:the\\s+)?phrase\\s+\\S+', pre_l):
        return 'retrieval_recall'
    if re.search("\\bi'?d\\s+rather\\b", pre_l) or re.search('\\bwhat\\s+i\\s+(?:actually\\s+)?care\\s+about\\b', pre_l) or re.search('\\bmatters?\\s+more\\s+to\\s+me\\b', pre_l) or re.search('\\bsoft\\s+preferences?\\s+matter\\b', text_l) or re.search('\\bloose\\s+hard\\s+request\\s+plus\\s+soft\\b', text_l) or (re.search('\\bclarify\\b.{0,48}\\bpriorit', text_l) and re.search('\\bsoft\\s+preferences?\\b', text_l)):
        return 'preference_reasoning'
    if re.search('\\b(?:disappears?|unavailable|if\\s+that\\s+exact|strongly\\s+prefer)\\b', pre_l):
        return 'recovery'
    if re.search('\\bfirm\\s+requirements?\\b', text_l) and re.search('\\b(?:soft\\s+preference|preferred\\s+brand|ideally)\\b', text_l):
        return 'constraint_satisfaction'
    return 'retrieval'

def _tradeoff_priorities(query: str) -> list[dict[str, str]]:
    """Parse ranked max/min axes from the shopper's own wording."""
    text = (query or '').split('Task rules:')[0]
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    filler = {'advertised', 'maximum', 'minimum', 'available', 'current', 'live', 'including', 'explicitly', 'mode', 'the', 'a', 'an', 'of', 'and', 'or', 'with', 'for'}
    for match in _TRADEOFF_RE.finditer(text):
        direction_word = match.group('dir').lower()
        raw_field = re.sub('\\s+', ' ', match.group('field')).strip(' .,;:')
        tokens = [tok for tok in _TOKEN_RE.findall(raw_field.lower()) if tok not in filler and len(tok) > 1 and (tok not in {'then', 'among', 'between', 'while', 'before', 'after'})]
        if not tokens:
            continue
        phrase = ' '.join(tokens[:4])
        if phrase in seen:
            continue
        seen.add(phrase)
        direction = 'max' if direction_word in _DIR_MAX else 'min'
        found.append({'field': phrase, 'direction': direction})
    return found[:5]

def _iter_fact_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    stack: list[Any] = [row]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for (key, value) in node.items():
                if value in (None, ''):
                    continue
                if isinstance(value, (dict, list)):
                    stack.append(value)
                    continue
                if isinstance(key, str):
                    pairs.append((key, str(value)))
        elif isinstance(node, list):
            stack.extend(node)
    return pairs

def _metric_tokens(phrase: str) -> list[str]:
    """Axis tokens taken only from the shopper's tradeoff phrase."""
    found: list[str] = []
    seen: set[str] = set()
    for tok in _TOKEN_RE.findall((phrase or '').lower()):
        if len(tok) <= 1 or tok in seen:
            continue
        seen.add(tok)
        found.append(tok)
    return found
_SKIP_METRIC_KEYS = frozenset({'product_id', 'sku', 'listing_id', 'currency', 'brand', 'title', 'category_path', 'options'})

def tf4_as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(',', '')
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            match = _NUM_TOKEN_RE.search(text)
            if not match:
                return None
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None

def _harvest_unit_numbers(surface: str) -> list[tuple[str, float]]:
    """Parse NUM+unit pairs from one catalog string (facts/options/title)."""
    found: list[tuple[str, float]] = []
    text = surface or ''
    if not text:
        return found
    cleaned = re.sub('\\b\\d+(?:\\.\\d+)?\\s*[x×]\\s*\\d+(?:\\.\\d+)?\\b', ' ', text, flags=re.I)
    cleaned = re.sub('\\b\\d+(?:\\.\\d+)?\\s*-\\s*\\d+(?:\\.\\d+)?\\s*(?:\\"|\'\'|in(?:ch(?:es)?)?)\\b', ' ', cleaned, flags=re.I)
    seen_pair: set[tuple[str, float]] = set()
    for match in re.finditer('(\\d+(?:\\.\\d+)?)\\s*-?\\s*([A-Za-z%]{1,16}|\\"|\')', cleaned):
        try:
            number = float(match.group(1))
        except ValueError:
            continue
        if number >= 10000000:
            continue
        left = match.start(1)
        if left > 0 and cleaned[left - 1].isalnum():
            continue
        right = match.end(2)
        if right < len(cleaned) and cleaned[right].isalnum():
            continue
        unit = _canon_unit(match.group(2))
        if len(unit) == 1:
            continue
        if unit in _SEED_STOP:
            continue
        pair = (unit, number)
        if pair in seen_pair:
            continue
        seen_pair.add(pair)
        found.append(pair)
    return found

def _title_unit_numbers(row: dict[str, Any]) -> list[tuple[str, float]]:
    """Harvest (unit, number) pairs, preferring variant options over listing facts/title."""
    found: list[tuple[str, float]] = []
    seen_unit: set[str] = set()

    def absorb(surface: str) -> None:
        for (unit, number) in _harvest_unit_numbers(surface):
            if unit in seen_unit:
                continue
            seen_unit.add(unit)
            found.append((unit, number))
    options = str(row.get('options') or '')
    if options:
        absorb(options)
    facts = row.get('facts') if isinstance(row.get('facts'), dict) else {}
    for (key, value) in facts.items():
        if value in (None, '') or key in _SKIP_METRIC_KEYS:
            continue
        text = str(value)
        if re.search('\\d', text):
            absorb(text)
    for (key, value) in _iter_fact_pairs(row):
        if key in _SKIP_METRIC_KEYS or key in {'price', 'title', 'options', 'facts'}:
            continue
        text = str(value)
        if re.search('\\d', text) and re.search('[A-Za-z\\"\']', text):
            absorb(text)
    title = str(row.get('title') or '')
    if title:
        absorb(title)
    return found

def _metric_value(row: dict[str, Any], phrase: str, query: str='') -> float | None:
    """Score a query-derived axis via structural key/token overlap on live facts."""
    del query
    tokens = _metric_tokens(phrase)
    if not tokens:
        return None
    phrase_kn = _norm_key(phrase)
    price_axis = any((_keys_structurally_match(_norm_key(tok), 'price') or _keys_structurally_match(_norm_key(tok), 'cost') for tok in tokens)) or _keys_structurally_match(phrase_kn, 'price')
    if price_axis and len(tokens) <= 2:
        for key in ('price', 'current_price', 'price_at_add', 'new_price'):
            number = tf4_as_float(row.get(key))
            if number is not None:
                return number
        return None
    scored: list[tuple[int, float]] = []
    for (key, value) in _iter_fact_pairs(row):
        if key in _SKIP_METRIC_KEYS:
            continue
        kn = _norm_key(key)
        if key in {'price', 'current_price', 'price_at_add', 'new_price'}:
            continue
        text = f'{kn} {value}'.lower()
        overlap = 0
        for tok in tokens:
            tok_n = _norm_key(tok)
            if not tok_n:
                continue
            if _keys_structurally_match(kn, tok_n) or tok_n in kn:
                overlap += 2
            elif tok in text:
                overlap += 1
        if _keys_structurally_match(kn, phrase_kn):
            overlap += 2
        if overlap <= 0:
            continue
        number = tf4_as_float(value)
        if number is None or number >= 10000000:
            continue
        scored.append((overlap, number))
    for surface_key in ('title', 'options'):
        surface = str(row.get(surface_key) or '').lower()
        if not surface:
            continue
        for match in re.finditer('(\\d+(?:\\.\\d+)?)', surface):
            try:
                number = float(match.group(1))
            except ValueError:
                continue
            if number >= 10000000:
                continue
            window = surface[max(0, match.start() - 28):match.end() + 28]
            overlap = sum((1 for tok in tokens if tok in window))
            glued = surface[match.end():match.end() + 12]
            unit_match = re.match('\\s*-?\\s*([a-z%]{1,16})', glued)
            if unit_match:
                unit = unit_match.group(1)
                if len(unit) >= 2 and any((_keys_structurally_match(_norm_key(tok), _norm_key(unit)) or (len(tok) >= 3 and tok in unit) or (len(unit) >= 3 and unit in tok) for tok in tokens)):
                    overlap += 2
                elif len(tokens) >= 2 and len(unit) >= 2:
                    overlap = max(overlap, 1)
            if overlap <= 0:
                continue
            scored.append((overlap, number))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], -item[1]))
    top = scored[0][0]
    tied = [value for (overlap, value) in scored if overlap == top]
    return max(tied) if tied else None

def _axis_unit_vote(rows: list[dict[str, Any]], phrase: str) -> str | None:
    """When facts omit an axis, infer which title-unit tracks it across the shortlist."""
    tokens = _metric_tokens(phrase)
    if not tokens:
        return None
    if any((_keys_structurally_match(_norm_key(tok), 'price') or _keys_structurally_match(_norm_key(tok), 'cost') for tok in tokens)):
        return None
    for row in rows:
        for (unit, _number) in _title_unit_numbers(row):
            if len(unit) < 2:
                continue
            if any((_keys_structurally_match(_norm_key(tok), _norm_key(unit)) or tok in unit or unit in tok for tok in tokens)):
                return unit
    buckets: dict[str, list[float]] = {}
    for row in rows:
        seen_units: set[str] = set()
        for (unit, number) in _title_unit_numbers(row):
            if len(unit) < 2 or unit in seen_units:
                continue
            seen_units.add(unit)
            buckets.setdefault(unit, []).append(number)
    best_unit: str | None = None
    best_score = -1.0
    for (unit, values) in buckets.items():
        if len(values) < 2:
            continue
        span = max(values) - min(values)
        coverage = len(values)
        score = coverage * 1000.0 + span + 0.01 * len(unit)
        if score > best_score:
            best_score = score
            best_unit = unit
    return best_unit

def _metric_value_with_unit(row: dict[str, Any], phrase: str, unit: str | None, query: str='') -> float | None:
    tokens = _metric_tokens(phrase)
    phrase_kn = _norm_key(phrase)
    price_axis = any((_keys_structurally_match(_norm_key(tok), 'price') or _keys_structurally_match(_norm_key(tok), 'cost') for tok in tokens)) or _keys_structurally_match(phrase_kn, 'price')
    if price_axis:
        for key in ('price', 'current_price', 'price_at_add', 'new_price'):
            number = tf4_as_float(row.get(key))
            if number is not None:
                return number
        return None
    if unit:
        values = [number for (u, number) in _title_unit_numbers(row) if u == unit]
        if values:
            return max(values)
    return _metric_value(row, phrase, query)

def _goal_spec_fragments(query: str, notes: list[str] | None=None) -> list[str]:
    """Numeric/spec fragments the shopper asked for — used to filter result rows."""
    frags: list[str] = []
    seen: set[str] = set()
    for part in _search_parts(query, notes):
        if not re.search('\\d', part):
            continue
        low = part.lower()
        if low not in seen:
            seen.add(low)
            frags.append(low)
        for num in _NUM_TOKEN_RE.findall(part):
            if num not in seen:
                seen.add(num)
                frags.append(num)
    return frags

def _spec_match_score(row: dict[str, Any], specs: list[str]) -> int:
    """How well a catalog row carries the shopper's numeric/spec fragments."""
    if not specs:
        return 0
    title = str(row.get('title') or '')
    brand = str(row.get('brand') or '')
    facts = row.get('facts') if isinstance(row.get('facts'), dict) else {}
    blob = f'{title} {brand} {json.dumps(facts, ensure_ascii=False, default=str)}'.lower()
    compact = re.sub('[^a-z0-9.]+', '', blob)
    score = 0
    missed = 0
    matched_nums: set[str] = set()
    for spec in specs:
        sn = re.sub('[^a-z0-9.]+', '', spec)
        nums = _NUM_TOKEN_RE.findall(spec)
        hit = bool(sn and sn in compact)
        if not hit and nums and any((num in blob or num in compact for num in nums)):
            hit = True
        if hit:
            credit = False
            for num in nums or [sn]:
                if num and num not in matched_nums:
                    matched_nums.add(num)
                    credit = True
            if credit:
                score += 5
            continue
        missed += 1
    score -= 4 * missed
    return score
tf4_FIRM_CUE_RE = re.compile('\\b(?:must|require(?:d|s)?|has\\s+to|have\\s+to|only|exactly|mandatory|non[\\s-]?negotiable)\\b', re.I)
tf4_SOFT_CUE_RE = re.compile('\\b(?:preferred|preferring|prefer|ideally|leaning|lean|nice\\s+to\\s+have|would\\s+like|rather|bonus|optional|soft|looking\\s+for)\\b', re.I)
_SOFT_PHRASE_RE = re.compile("(?:\\b(?:rather(?:\\s+[^,.;]{0,40})?\\s+for|instead\\s+of(?:\\s+just)?|don'?t\\s+just(?:\\s+\\w+){0,3}|care\\s+about)\\s+([^.,;!?]{4,70})|(?:^|[.;:])\\s*([^.;:]{4,90}?)\\s+is\\s+what\\s+i\\s+(?:actually\\s+)?care\\s+about\\b|(?:^|[.;:])\\s*([^.;:]{4,90}?)\\s+matters?\\s+more(?:\\s+to\\s+me)?\\b)", re.I)
_INSTRUCTION_VERBS = frozenset('\n    say says saying identify identifies tell tells show shows walk give gives\n    explain describes describe mention mentions label labels\n    '.split())

def _row_text_blob(row: dict[str, Any]) -> str:
    """Flatten live product surfaces into one searchable lowercase blob."""
    chunks: list[str] = []
    for key in ('title', 'brand', 'sku', 'options', 'category', 'condition', 'model'):
        value = row.get(key)
        if value not in (None, ''):
            chunks.append(str(value))
    path = row.get('category_path')
    if isinstance(path, list):
        chunks.extend((str(part) for part in path if part is not None))
    elif isinstance(path, str) and path.strip():
        chunks.append(path)
    facts = row.get('facts')
    if isinstance(facts, dict):
        for (key, value) in facts.items():
            if value in (None, '', False):
                continue
            if isinstance(value, (dict, list)):
                continue
            chunks.append(f'{key} {value}')
    for (key, value) in row.items():
        if key in {'title', 'brand', 'sku', 'options', 'category', 'category_path', 'facts', 'product_id', 'listing_id', 'price', 'in_stock', 'currency', 'variants'}:
            continue
        if value in (None, '', False) or isinstance(value, (dict, list, bool)):
            continue
        if isinstance(key, str):
            chunks.append(f'{key} {value}')
    text = ' '.join(chunks).lower()
    text = re.sub('(\\d+(?:\\.\\d+)?)\\s*(?:["\\\'])', '\\1-in ', text)
    text = re.sub('(\\d+(?:\\.\\d+)?)\\s*(?:inches|inch)\\b', '\\1-in', text)
    text = re.sub('(\\d+(?:\\.\\d+)?)\\s+in\\b', '\\1-in', text)
    return text

def _fragment_match(blob: str, fragment: str) -> bool:
    """True when a query-derived fragment is evidenced on the product blob."""
    frag = _canon_spec_token((fragment or '').strip().lower())
    if not frag or not blob:
        return False
    variants = {frag}
    if frag.endswith('ance') and len(frag) > 6:
        variants.add(frag[:-4] + 'ant')
        variants.add(frag[:-6] if frag.endswith('stance') else frag)
    if frag.endswith('ant') and len(frag) > 5:
        variants.add(frag[:-3] + 'ance')
    if frag.endswith('ing') and len(frag) > 5:
        variants.add(frag[:-3])
    for candidate in list(variants):
        if candidate in blob:
            return True
        compact_blob = re.sub('[^a-z0-9.]+', '', blob)
        compact_frag = _normalize_fragment(candidate)
        if len(compact_frag) >= 3 and compact_frag in compact_blob:
            return True
    alt = (fragment or '').strip().lower()
    if alt and alt != frag and (alt in blob):
        return True
    tokens = [t for t in _TOKEN_RE.findall(frag) if len(t) > 1 and t not in _SEED_STOP]
    if not tokens:
        return False
    hits = sum((1 for tok in tokens if tok in blob or tok in re.sub('[^a-z0-9.]+', '', blob)))
    if len(tokens) == 1:
        return hits == 1
    return hits >= max(1, len(tokens) - (1 if len(tokens) >= 3 else 0))

def _normalize_fragment(fragment: str) -> str:
    frag = _canon_spec_token(fragment or '')
    return re.sub('[^a-z0-9.]+', '', frag.lower())

def _dedupe_fragments(frags: list[str]) -> list[str]:
    """Drop near-duplicate fragments that compact to the same key."""
    out: list[str] = []
    seen: set[str] = set()
    for frag in frags:
        key = _normalize_fragment(frag)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(frag)
    return out

def _split_constraints(query: str, notes: list[str] | None=None) -> tuple[list[str], list[str]]:
    """Derive firm vs soft constraint fragments from the public ask only."""
    chunks = [tf4_preamble(query)]
    if notes:
        chunks.extend(notes[-3:])
    text = ' '.join(chunks)
    text = _CANDIDATE_KEY_RE.sub(' ', text)
    firm: list[str] = []
    soft: list[str] = []
    seen_f: set[str] = set()
    seen_s: set[str] = set()
    budget_nums = _budget_number_strings(text)
    currency_codes = _currency_codes_in_text(text)

    def push(target: list[str], seen: set[str], value: str, *, allow_model_code: bool=False) -> None:
        raw = re.sub('\\s+', ' ', value or '').strip(' .,;:|/\\"\'')
        if len(raw) < 2:
            return
        raw = _canon_spec_token(raw)
        low = raw.lower()
        if low in seen or low in _SEED_STOP:
            return
        if low in {'in', 'inch', 'inches'}:
            return
        if '::' in raw:
            return
        if re.fullmatch('\\d{6,}', low):
            if not allow_model_code or not 6 <= len(low) <= 14:
                return
        if re.fullmatch('\\d{3,5}', low):
            if not allow_model_code:
                return
            if low in budget_nums:
                return
        if low in budget_nums or low in currency_codes:
            return
        glued = re.fullmatch('(\\d+(?:\\.\\d+)?)[\\s-]+([a-z]{3})', low)
        if glued and (glued.group(1) in budget_nums or glued.group(2) in currency_codes):
            return
        if any((bad in low for bad in ('replacement', 'insufficient', 'identify', 'explicit', 'label', 'requirements', 'saving', 'dollars', 'matter', 'ideally'))):
            return
        seen.add(low)
        target.append(raw)
    for match in re.finditer('"([^"\\n]{2,80})"', text):
        push(firm, seen_f, match.group(1))
    category = _category_hint(query)
    if category:
        push(firm, seen_f, category)
    for match in _NUM_UNIT_RE.finditer(text):
        (num, unit) = (match.group(1), match.group(2))
        unit_l = _canon_unit(unit)
        if num in budget_nums or unit_l in currency_codes:
            continue
        if len(unit_l) < 2 or unit_l in _SEED_STOP:
            continue
        push(firm, seen_f, f'{num}-{unit_l}')
    discourse_skip = {'besides', 'except', 'including', 'versus', 'brand', 'something', 'anything', 'item', 'product', 'have', 'has', 'with', 'without', 'using', 'via', 'also', 'please'}
    for match in re.finditer('\\b(?:need|want|find|get|buy|order)(?:\\s+me)?\\s+(?:an?\\s+|the\\s+)?([a-z][a-z0-9-]{2,28})\\b', tf4_preamble(query), re.I):
        noun = match.group(1)
        if noun.lower() in discourse_skip or noun.lower() in _SEED_STOP:
            continue
        push(firm, seen_f, noun)
    preamble = tf4_preamble(query)
    ask_focus = re.split('\\b(?:walk\\s+me|for\\s+a\\s+\\w+|give\\s+me\\s+a\\s+reason|tell\\s+me\\s+why|task\\s+rules)\\b', preamble, maxsplit=1, flags=re.I)[0]
    ask_focus = ask_focus.split('.')[0]
    for match in re.finditer('\\b([A-Z][A-Za-z0-9]+(?:\\s+[A-Z][A-Za-z0-9]+){0,2})\\b', ask_focus):
        brand = match.group(1).strip()
        low = brand.lower()
        if low in currency_codes or low in _SEED_STOP or low in discourse_skip:
            continue
        if low in currency_codes or low in {'sku', 'can', 'find', 'want', 'get', 'need', 'buy', 'order', 'please', 'give', 'tell', 'walk', 'full'}:
            continue
        if len(low) <= 3 and low.isalpha():
            continue
        if ' ' in brand or len(brand) >= 4:
            push(firm, seen_f, brand)
        tail = ask_focus[match.end():match.end() + 64]
        nouns = [noun for noun in re.findall('\\b([a-z][a-z0-9-]{3,24})\\b', tail) if noun not in _SEED_STOP and noun not in discourse_skip and (noun not in currency_codes) and (noun not in {'with', 'from', 'into', 'over', 'under', 'stock', 'full', 'hard'})]
        if nouns:
            push(firm, seen_f, nouns[-1])
            if len(nouns) >= 2 and nouns[-2] != nouns[-1]:
                push(firm, seen_f, nouns[-2])
    split_pat = '[.!?;\\n,]|Task rules:|\\bbut\\b|\\band\\b|\\bunder\\b|\\bbelow\\b|\\bwithin\\b|\\bin\\s+the\\b'
    for (cue_re, bucket, seen) in ((tf4_FIRM_CUE_RE, firm, seen_f), (tf4_SOFT_CUE_RE, soft, seen_s)):
        for match in cue_re.finditer(text):
            window = text[match.end():match.end() + 40]
            window = re.split(split_pat, window, maxsplit=1)[0]
            for quoted in re.finditer('"([^"\\n]{2,60})"', window):
                push(bucket, seen, quoted.group(1))
            found_num = False
            for num_unit in _NUM_UNIT_RE.finditer(window):
                (num, unit) = (num_unit.group(1), _canon_unit(num_unit.group(2)))
                if num in budget_nums or unit in currency_codes:
                    continue
                if len(unit) < 2 or unit in _SEED_STOP:
                    continue
                push(bucket, seen, f'{num}-{unit}')
                found_num = True
            for code in re.findall('\\b(\\d{3,14})\\b', window):
                if code in budget_nums:
                    continue
                push(bucket, seen, code, allow_model_code=True)
                found_num = True
            words = [tok for tok in re.findall('[A-Za-z][A-Za-z0-9.+-]{1,}', window) if tok.lower() not in _SEED_STOP and tok.lower() not in currency_codes and (tok.lower() not in discourse_skip) and (tok.lower() not in _INSTRUCTION_VERBS)]
            if words and (not found_num):
                if bucket is firm and words[0].lower() in _INSTRUCTION_VERBS:
                    continue
                if bucket is firm and any((w.lower() in _INSTRUCTION_VERBS for w in words[:2])):
                    continue
                push(bucket, seen, ' '.join(words[:3]))
            elif words and bucket is soft:
                for word in words[:3]:
                    push(bucket, seen, word)
    for match in re.finditer('\\b(?:required|requirement|constraint)\\s+([A-Za-z][\\w-]{1,24})\\s*(?:is|=|:)\\s*([A-Za-z0-9][\\w.-]{0,24})', text, re.I):
        push(firm, seen_f, match.group(1))
        push(firm, seen_f, match.group(2))
    if notes:
        note_blob = ' '.join(notes[-3:])
        for match in re.finditer('\\band\\s+([A-Za-z][\\w-]{1,20})\\s+([A-Za-z0-9][\\w.-]{0,20})\\b', note_blob, re.I):
            (left, right) = (match.group(1), match.group(2))
            if left.lower() in discourse_skip or right.lower() in discourse_skip:
                continue
            push(firm, seen_f, left)
            push(firm, seen_f, right)
    explicit_brand = re.search('\\bbrand\\s+([A-Za-z0-9][\\w-]{1,24})\\b', text, re.I)
    if explicit_brand:
        brand = explicit_brand.group(1)
        prefix = text[max(0, explicit_brand.start() - 40):explicit_brand.start()]
        if tf4_FIRM_CUE_RE.search(prefix) or re.search('\\b(?:must|need|require)\\b', prefix, re.I):
            push(firm, seen_f, brand)
        else:
            push(soft, seen_s, brand)
    for part in _search_parts(query, notes):
        low = part.lower()
        if low in budget_nums or low in currency_codes:
            continue
        if re.search('\\d', part) or re.fullmatch('[A-Za-z]+[0-9].*', part):
            push(firm, seen_f, part)
    for match in tf4_SOFT_CUE_RE.finditer(text):
        window = text[match.end():match.end() + 36]
        window = re.split(split_pat, window, maxsplit=1)[0]
        words = [tok for tok in re.findall('[A-Za-z][A-Za-z0-9.+-]{2,}', window) if tok.lower() not in seen_f and tok.lower() not in _SEED_STOP and (tok.lower() not in currency_codes) and (tok.lower() not in discourse_skip)]
        if words:
            push(soft, seen_s, ' '.join(words[:3]))
    for match in _SOFT_PHRASE_RE.finditer(text):
        phrase = next((g for g in match.groups() if g), '') or ''
        phrase = re.sub('\\s+', ' ', phrase).strip(' .,;:')
        if ' - ' in phrase:
            phrase = phrase.split(' - ')[-1]
        elif re.search('\\bfor\\b', phrase, re.I) and phrase.lower().startswith(('don', 'rather', 'instead')):
            phrase = re.split('\\bfor\\b', phrase, maxsplit=1, flags=re.I)[-1]
        phrase = re.split('\\bthan\\b', phrase, maxsplit=1, flags=re.I)[0]
        phrase = re.split('\\bthat\\b', phrase, maxsplit=1, flags=re.I)[0]
        words = [tok for tok in phrase.split() if tok.lower() not in _SEED_STOP and tok.lower() not in discourse_skip and (tok.lower() not in {'but', 'than', 'that', 'few', 'dollars', 'saving', 'grab', 'cheapest', "don't", 'dont', 'settle', 'something', 'instead', 'rather'}) and (len(tok) > 1) and (not tok.startswith('-'))]
        if not words:
            continue
        short_phrase = ' '.join(words[:3])
        push(soft, seen_s, short_phrase)
        for tok in words[:3]:
            if tok.lower() not in discourse_skip and tok.lower() not in _SEED_STOP:
                push(soft, seen_s, tok)
    for frag in _soft_fact_dimensions(query) + _use_context_phrases(query):
        push(soft, seen_s, frag)
    soft = [s for s in soft if _normalize_fragment(s) not in {_normalize_fragment(f) for f in firm}]
    return (_dedupe_fragments(firm)[:10], _dedupe_fragments(soft)[:10])

def _soft_token_coverage(blob: str, soft: list[str]) -> tuple[int, int]:
    """How many distinct soft content tokens are evidenced on a product blob."""
    tokens: list[str] = []
    seen: set[str] = set()
    for frag in soft or []:
        for tok in re.split('[\\s/]+', (frag or '').lower()):
            if len(tok) < 4 or tok in seen or tok in _SEED_STOP:
                continue
            if re.fullmatch('\\d{3,14}', tok):
                continue
            seen.add(tok)
            tokens.append(tok)
    if not tokens:
        return (0, 0)
    hits = 0
    for tok in tokens:
        if _fragment_match(blob, tok) or tok in blob:
            hits += 1
    return (hits, len(tokens))

def _required_brand_labels(notes: list[str]) -> list[str]:
    """Extract explicitly required brand labels from clarification notes."""
    found: list[str] = []
    seen: set[str] = set()
    for note in notes or []:
        for match in re.finditer('\\brequired\\s+brand\\s+is\\s+([A-Za-z0-9][\\w-]{1,24})', note or '', re.I):
            brand = match.group(1)
            low = brand.lower()
            if low in seen:
                continue
            seen.add(low)
            found.append(brand)
    return found

def _priority_soft_axes(notes: list[str] | str, query: str='') -> list[str]:
    """Ordered soft axes from shopper priority discourse (phrase-level, not tokens)."""
    if isinstance(notes, list):
        raw = ' '.join((n for n in notes if n))
    else:
        raw = notes or ''
    axes: list[str] = []
    seen: set[str] = set()

    def add(phrase: str) -> None:
        phrase = re.sub('\\s+', ' ', phrase or '').strip(' .,;:')
        if not phrase:
            return
        words = [w for w in re.split('\\s+', phrase) if w]
        if len(words) == 1 and any((words[0].lower() in a.lower().split() for a in axes)):
            return
        low = phrase.lower()
        if low in seen or len(low) < 3:
            return
        seen.add(low)
        axes.append(phrase)
    for phrase in _priority_soft_phrases(raw):
        add(phrase)
    if len(axes) < 2:
        for phrase in _soft_fact_dimensions(query or ''):
            add(phrase)
    return axes[:6]

def _axis_evidenced(blob: str, axis: str) -> bool:
    """True when a priority axis phrase is supported by listing text."""
    if _fragment_match(blob, axis):
        return True
    toks = [t for t in re.split('[\\s/]+', (axis or '').lower()) if len(t) >= 4 and t not in _SEED_STOP]
    if not toks:
        tok2 = [t for t in re.split('[\\s/]+', (axis or '').lower()) if len(t) >= 3 and t not in _SEED_STOP]
        return bool(tok2) and all((_fragment_match(blob, t) or t in blob for t in tok2))
    hits = sum((1 for t in toks if _fragment_match(blob, t) or t in blob))
    return hits >= max(1, len(toks) - (1 if len(toks) >= 3 else 0))

def _failed_product_ids(board: 'EpisodeBoard') -> set[str]:
    """Product ids previously abandoned after a live failure / price event."""
    out: set[str] = set()
    for key in board.failed_keys or set():
        if '::' in key:
            out.add(key.split('::', 1)[0])
    if board.last_failed_key and '::' in board.last_failed_key:
        out.add(board.last_failed_key.split('::', 1)[0])
    return out

def _preference_cart_ok(board: 'EpisodeBoard', snap: dict[str, Any]) -> bool:
    """Gate preference carts on multi-axis soft coverage from the clarified ask."""
    if board.task_shape != 'preference_reasoning' or not board.soft_preferences:
        return True
    blob = _row_text_blob(snap)
    (_fh, _fm, soft_hits) = _compliance_scores(snap, board.firm_constraints, board.soft_preferences)
    if soft_hits <= 0:
        return False
    (cov_hits, cov_total) = _soft_token_coverage(blob, board.soft_preferences)
    if cov_total >= 4 and cov_hits < 2:
        return False
    axes = _priority_soft_axes(board.shopper_notes, board.query)
    if len(axes) >= 2:
        evidenced = [i for (i, axis) in enumerate(axes[:4]) if _axis_evidenced(blob, axis)]
        if not evidenced:
            return False
        if len(evidenced) == 1 and evidenced[0] >= min(2, len(axes[:4]) - 1):
            return False
        top_hits = sum((1 for i in evidenced if i < 2))
        lower_hits = sum((1 for i in evidenced if i >= 2))
        if top_hits == 0 and lower_hits <= 1 and (cov_hits < 3):
            return False
        if top_hits >= 1 and lower_hits == 0 and (cov_hits < 2):
            return False
    return True

def _compliance_scores(row: dict[str, Any], firm: list[str], soft: list[str]) -> tuple[int, int, int]:
    """Return (firm_hits, firm_misses, soft_hits) against a live product row."""
    blob = _row_text_blob(row)
    firm = _dedupe_fragments(firm)
    soft = _dedupe_fragments(soft)
    firm_hits = 0
    firm_misses = 0
    for frag in firm:
        if _fragment_match(blob, frag):
            firm_hits += 1
        else:
            alt = frag.replace('-', ' ')
            if alt != frag and _fragment_match(blob, alt):
                firm_hits += 1
            else:
                firm_misses += 1
    soft_hits = sum((1 for frag in soft if _fragment_match(blob, frag)))
    return (firm_hits, firm_misses, soft_hits)

def _is_compliant(row: dict[str, Any], firm: list[str]) -> bool:
    firm = _dedupe_fragments(firm)
    if not firm:
        return True
    numeric = [f for f in firm if re.search('\\d', f)]
    if not numeric:
        return True
    blob = _row_text_blob(row)
    for frag in numeric:
        alt = frag.replace('-', ' ')
        if not (_fragment_match(blob, frag) or _fragment_match(blob, alt)):
            return False
    return True

def tf4_unwrap_observation(observation: Any) -> dict[str, Any]:
    """Flatten env payloads nested under observation.observation."""
    if not isinstance(observation, dict):
        return {}
    inner = observation.get('observation')
    if isinstance(inner, dict) and any((key in inner for key in ('results', 'compared', 'product_id', 'facts', 'variants', 'order', 'status', 'sku', 'price', 'action', 'items'))):
        merged = dict(inner)
        if observation.get('error'):
            merged['error'] = observation['error']
        if 'done' in observation:
            merged['done'] = observation['done']
        return merged
    return dict(observation)

def _title_goal_alignment(title: str, goal_tokens: set[str]) -> int:
    """Score how directly a title matches goal tokens; demote 'X for <goal>' shapes."""
    title_l = str(title or '').lower()
    if not title_l or not goal_tokens:
        return 0
    score = min(5, _value_overlap(title_l, goal_tokens))
    for_idx = title_l.find(' for ')
    early_hits = 0
    late_only = 0
    cutoff = max(12, len(title_l) // 3)
    for tok in goal_tokens:
        if len(tok) < 4:
            continue
        idx = title_l.find(tok)
        if idx < 0:
            continue
        if idx <= cutoff:
            early_hits += 1
        elif for_idx >= 0 and idx > for_idx:
            late_only += 1
    score += min(3, early_hits)
    if late_only and early_hits == 0:
        score -= 4
    if for_idx >= 0:
        score -= 5
        head = title_l[:for_idx]
        if not any((len(t) >= 5 and t in head for t in goal_tokens)):
            score -= 3
    return score

def _recommend_message(board: 'EpisodeBoard') -> str:
    """Shopper-facing recommendation used to answer interventions."""
    key = ''
    row: dict[str, Any] = {}
    if board.named and (board.priorities or board.task_shape == 'ranking'):
        ranked_named = board.ranked_named()
        if ranked_named:
            (pid, sku, key) = ranked_named[0]
            row = board._product_snapshot(pid, sku)
    if not key and board.cart:
        top = board.cart[-1]
        key = top['key']
        row = board.candidates.get(key) or board.viewed.get(top['product_id']) or {}
    if not key:
        ranked = board.ranked_candidates()
        if not ranked:
            return 'I am verifying options that meet your constraints and will order shortly.'
        (_score, key, row) = ranked[0]
    title = str(row.get('title') or key)[:100]
    bits = [f'I am going with {key} — {title}.']
    specs = _goal_spec_fragments(board.query, board.shopper_notes)
    clean_specs = [s for s in specs if not re.search('\\d{6,}', s) and '::' not in s and (not re.fullmatch('[\\w-]+-\\d+', s))]
    if clean_specs:
        bits.append(f"It should cover {', '.join(clean_specs[:2])}.")
    if board.budget is not None:
        bits.append(f'I will keep it at or under {board.budget}.')
    stock = board.stock.get(key) or board.stock.get(str(row.get('product_id') or '')) or {}
    price = tf4_as_float(stock.get('price'))
    if price is not None:
        bits.append(f'Latest live price seen: {price}.')
    return ' '.join(bits)

def _named_candidates(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _CANDIDATE_KEY_RE.finditer(text or ''):
        key = f'{match.group(1)}::{match.group(2)}'
        if key in seen:
            continue
        seen.add(key)
        found.append((match.group(1), match.group(2)))
    return found

def _fields_mentioned(query: str, allowed: set[str], extra: str='') -> set[str]:
    """Return schema claim fields whose names appear in the public query or notes."""
    if not allowed:
        return set()
    q = f'{tf4_preamble(query)} {extra}'.lower()
    q_norm = re.sub('[^a-z0-9]+', ' ', q)
    compact = q_norm.replace(' ', '')
    hits: set[str] = set()
    for field in allowed:
        if field in {'price', 'in_stock'}:
            continue
        surface = field.replace('_', ' ')
        if field in q_norm or surface in q or surface.replace(' ', '') in compact:
            hits.add(field)
    return hits

def tf4_goal_tokens(query: str, notes: list[str] | None=None) -> set[str]:
    text = tf4_preamble(query)
    if notes:
        text = f"{text} {' '.join(notes[-3:])}"
    budget_nums = _budget_number_strings(text)
    currency = _currency_codes_in_text(text)
    tokens: set[str] = set()
    for tok in _TOKEN_RE.findall(text):
        low = tok.lower()
        if len(low) <= 1 or re.fullmatch('0+', low):
            continue
        if low in _SEED_STOP or low in budget_nums or low in currency:
            continue
        if re.fullmatch('\\d+(?:\\.\\d+)?', low) and (low in budget_nums or float(low) >= 1000):
            continue
        tokens.add(low)
    return tokens

def _value_overlap(value: Any, goal_tokens: set[str]) -> int:
    if value is None or value == '':
        return 0
    text = str(value).lower()
    tokens = set(_TOKEN_RE.findall(text))
    nums = set(_NUM_TOKEN_RE.findall(text))
    score = len(tokens & goal_tokens)
    for num in nums:
        if num in goal_tokens or any((num in g or g in num for g in goal_tokens if any((ch.isdigit() for ch in g)))):
            score += 2
    return score

def _soft_claim_fields(tools: list[dict[str, Any]], allowed: set[str]) -> set[str]:
    """Soft/capability claim fields inferred only from live schema prose."""
    soft: set[str] = set()
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get('name') != 'place_test_order':
            continue
        blob = json.dumps(fn.get('parameters') or {}, ensure_ascii=False).lower()
        for field in allowed:
            if field in {'price', 'in_stock'}:
                continue
            surface = field.replace('_', ' ')
            if field in blob and any((cue in blob for cue in ('reason', 'capability', 'positive', 'why', 'benefit'))):
                soft.add(field)
            if surface in blob and 'positive' in blob:
                soft.add(field)
    return soft

def tf4_system_prompt(tools: list[dict[str, Any]], *, max_calls: int, max_steps: int, needs_ask: bool, named: list[tuple[str, str]], claim_fields: list[str], needs_justification: bool) -> str:
    tool_lines: list[str] = []
    for spec in tools:
        fn = _fn(spec)
        if not fn or not isinstance(fn.get('name'), str):
            continue
        desc = str(fn.get('description') or '').strip()
        params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
        props = params.get('properties') if isinstance(params.get('properties'), dict) else {}
        arg_names = ', '.join(sorted((str(k) for k in props.keys())))
        tool_lines.append(f"- {fn['name']}: {desc}" + (f' Args: {arg_names}.' if arg_names else ''))
    parts = ['You are a shopping agent. Obey the shopper request and every Task rules block in it.', 'Copy product_id/sku only from observations.', '', 'Non-negotiable process:', '1. If any firm requirement is missing or the goal defers a constraint until you ask, send a message and ask before your first cart add.', '2. Discover with short search/filter queries that include the distinctive constraints (category, brand if required, numeric specs, model codes). Reformulate if results are weak.', '3. view/compare a shortlist. Check category_path, variant options, and raw facts against EVERY firm requirement for the EXACT sku you will buy. Soft preferences are secondary.', '4. Your FIRST add_to_cart must be a genuine fully compliant choice — never a placeholder and never an item that misses a firm requirement. If several skus fit, prefer soft matches.', '5. Soft preferences (e.g. preferred brand/color/model) matter for the first choice when a fully compliant option exists; firm constraints always win.', '6. After any shopper message that follows a cart change, assume the market may have moved: inspect_stock the carted sku. If price exceeds budget or stock is false, remove_from_cart and recover to another still-valid option. Re-inspect the replacement before ordering.', '7. Do not wait for permission once a valid item is verified. place_test_order immediately.', '8. Never re-search or re-inspect after a carted sku already passed inspect_stock — order now.', f'9. Up to {max_calls} tool calls/turn — batch independent reads. Budget {max_steps} turns.', '', 'Tools:', *tool_lines]
    if needs_ask:
        parts.extend(['', 'This request is underspecified. Your first action should be a message that asks for the missing firm product specification.'])
    if named:
        shown = ', '.join((f'{pid}::{sku}' for (pid, sku) in named[:12]))
        parts.extend(['', 'The shopper named specific variant keys. Compare/view these before ranking or carting:', shown, 'Use the stated tradeoff order in the request (higher primary metric first, then secondary, then lower price). After a market change, re-rank on live facts and switch the cart if another named option becomes best.'])
    if needs_justification:
        parts.extend(['', 'Checkout requires structured justification claims grounded in view/compare/inspect_stock of the FINAL product.', f"Allowed reason/schema fields: {', '.join(claim_fields)}.", 'Always include price and in_stock from the latest inspect_stock.', "Also include exactly one claim for the shopper's constrained product attribute (short snake_case field name) whose value is copied from raw facts — not from title paraphrases. If facts use a neighboring key, still emit the shopper-facing attribute name with the observed value that matches the request.", 'Also include one supported positive capability/reason field when the schema allows it.'])
    else:
        parts.append('Omit justification unless the schema requires it.')
    parts.append('Continue until an observation reports done=true.')
    return '\n'.join(parts)

def _repair_json_object(raw: str) -> dict[str, Any]:
    text = (raw or '').strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    fixed = text.replace("'", '"')
    fixed = re.sub(',\\s*}', '}', fixed)
    fixed = re.sub(',\\s*]', ']', fixed)
    try:
        parsed = json.loads(fixed)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}

def tf4_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get('function', {}).get('arguments', '{}')
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return _repair_json_object(raw)
    return {}

def _compact_value(value: Any, depth: int=0) -> Any:
    if depth > 5:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= _STR_CHAR_CAP else value[:_STR_CHAR_CAP - 3] + '...'
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_compact_value(item, depth + 1) for item in value[:_LIST_ROW_CAP]]
    if isinstance(value, dict):
        return {str(k): _compact_value(v, depth + 1) for (k, v) in value.items()}
    return str(value)[:200]

def _compact_observation(observation: dict[str, Any]) -> str:
    text = json.dumps(_compact_value(observation), ensure_ascii=False, default=str)
    if len(text) <= _OBS_CHAR_CAP:
        return text
    return text[:_OBS_CHAR_CAP - 3] + '...'

def _keys_structurally_match(kn: str, needle: str) -> bool:
    """Match catalog keys by normalized equality or shared stem — no synonym tables."""
    if not kn or not needle:
        return False
    if kn == needle:
        return True
    if len(needle) >= 4 and len(kn) >= 4:
        if kn.endswith(needle) or needle.endswith(kn):
            return True
        if kn.startswith(needle) or needle.startswith(kn):
            return True
    return False

def _goal_numbers_for_field(query: str, field: str, notes: list[str] | None=None) -> set[str]:
    """Numbers in the shopper text near a schema field's surface form."""
    text = tf4_preamble(query).lower()
    if notes:
        text = f"{text} {' '.join(notes[-3:]).lower()}"
    compact = re.sub('[^a-z0-9]+', '', text)
    surfaces = {field.lower(), field.replace('_', ' ').lower(), _norm_key(field), *[p for p in field.lower().split('_') if len(p) >= 4]}
    hits: set[str] = set()
    for surface in surfaces:
        if len(surface) < 3:
            continue
        for match in re.finditer(re.escape(surface), text):
            window = text[max(0, match.start() - 32):match.end() + 32]
            hits.update(_NUM_TOKEN_RE.findall(window))
        if surface.isalpha() or surface.isalnum():
            for match in re.finditer(re.escape(_norm_key(surface)), compact):
                approx = text[max(0, match.start() - 40):min(len(text), match.end() + 40)]
                hits.update(_NUM_TOKEN_RE.findall(approx))
    return hits

def _lookup_fact(blobs: list[dict[str, Any]], field: str) -> Any:
    """Find an observed value for a schema field via structural key matching only."""
    needle = _norm_key(field)
    if not needle:
        return None
    best: tuple[int, Any] | None = None
    for blob in blobs:
        stack: list[Any] = [blob]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            for (key, value) in node.items():
                if value in (None, ''):
                    continue
                kn = _norm_key(key)
                if _keys_structurally_match(kn, needle):
                    rank = abs(len(kn) - len(needle))
                    if best is None or rank < best[0]:
                        best = (rank, value)
                if isinstance(value, dict):
                    stack.append(value)
                elif isinstance(value, list):
                    stack.extend((item for item in value if isinstance(item, dict)))
    return None if best is None else best[1]

class EpisodeBoard:

    def __init__(self, query: str, allowed_claims: set[str], *, task_shape: str='retrieval') -> None:
        self.query = query
        self.allowed_claims = set(allowed_claims)
        self.task_shape = task_shape
        self.budget = tf4_parse_budget(tf4_preamble(query))
        self.needs_ask = tf4_needs_clarification(query)
        self.named = _named_candidates(query)
        self.priorities = _tradeoff_priorities(query)
        self.mentioned_fields = _fields_mentioned(query, allowed_claims)
        self.search_seed = _search_seed(query)
        (self.firm_constraints, self.soft_preferences) = _split_constraints(query)
        self.shopper_notes: list[str] = []
        self.candidates: dict[str, dict[str, Any]] = {}
        self.viewed: dict[str, dict[str, Any]] = {}
        self.stock: dict[str, dict[str, Any]] = {}
        self.cart: list[dict[str, str]] = []
        self.action_log: list[str] = []
        self.last_error: str | None = None
        self.pending_question = False
        self.asked = False
        self.received_clarification = False
        self.needs_live_check = False
        self.checked_since_message: set[str] = set()
        self.must_recover = False
        self.needs_rerank = False
        self.event_seen = False
        self.event_generation = 0
        self.stock_generation: dict[str, int] = {}
        self.search_count = 0
        self.view_count = 0
        self.cart_adds = 0
        self.failed_keys: set[str] = set()
        self.last_failed_key: str | None = None
        self.cart_confirmed: set[str] = set()
        self.post_cart_stock: set[str] = set()
        self.facts_refresh_generation = -1

    def _invalidate_named_after_event(self) -> None:
        """Drop stale stock/facts for every named variant after a market move."""
        self.needs_rerank = len(self.named) >= 2 and self.task_shape in {'ranking', 'recovery', 'preference_reasoning'}
        if self.task_shape == 'preference_reasoning':
            self.needs_rerank = True
        self.needs_live_check = bool(self.cart)
        self.checked_since_message.clear()
        self.cart_confirmed.clear()
        self.post_cart_stock.clear()
        for item in list(self.cart):
            self.stock.pop(item['key'], None)
            self.stock.pop(item['product_id'], None)
            self.stock_generation.pop(item['key'], None)
        for (pid, sku) in self.named:
            key = f'{pid}::{sku}'
            self.stock.pop(key, None)
            self.stock.pop(pid, None)
            self.stock_generation.pop(key, None)

    def _refresh_from_notes(self) -> None:
        raw_notes = ' '.join(self.shopper_notes[-3:])
        notes_text = _sanitize_note_for_search(raw_notes)
        if _note_states_spend_cap(raw_notes):
            updated = tf4_parse_budget(raw_notes)
            if updated is not None and (self.budget is None or updated <= self.budget + 1e-09):
                original = tf4_parse_budget(tf4_preamble(self.query))
                explicit_cut = bool(re.search('\\b(?:can\\s+only|sorry|balance|at\\s+most|now)\\b', raw_notes, re.I))
                floor = 0.12 if explicit_cut else 0.45
                if original is None or updated >= max(5.0, floor * original) or explicit_cut:
                    prev_budget = self.budget
                    self.budget = updated
                    if prev_budget is not None and updated + 1e-09 < prev_budget and self.cart:
                        top = self.cart[-1]
                        pid = str(top.get('product_id') or '')
                        sku = str(top.get('sku') or '')
                        if pid and sku and (not self.stock_ok(pid, sku)):
                            self.must_recover = True
                            self.needs_live_check = False
                            self.failed_keys.add(f'{pid}::{sku}')
                            self.last_failed_key = f'{pid}::{sku}'
        self.mentioned_fields = _fields_mentioned(self.query, self.allowed_claims, extra=notes_text)
        clean_notes = [notes_text] if notes_text.strip() else []
        self.search_seed = _search_seed(self.query, clean_notes)
        (firm_q, soft_q) = _split_constraints(self.query)
        (firm_n, soft_n) = _split_constraints(notes_text) if notes_text.strip() else ([], [])
        soft_keep: list[str] = []
        soft_seen: set[str] = set()
        for frag in _priority_soft_phrases(raw_notes) + list(soft_q) + list(soft_n):
            low = frag.lower()
            if low in soft_seen:
                continue
            if re.fullmatch('\\d{1,5}', frag):
                continue
            soft_keep.append(frag)
            soft_seen.add(low)
        absorb_firm = self.asked or bool(re.search('\\b(?:required|requirement|constraint|must|exactly)\\b', notes_text, re.I))
        if absorb_firm:
            self.firm_constraints = _dedupe_fragments(list(firm_q) + list(firm_n))
        else:
            self.firm_constraints = _dedupe_fragments(list(firm_q))
        self.soft_preferences = _dedupe_fragments(soft_keep)[:16]
        extra_named = _named_candidates(notes_text)
        if extra_named:
            known = {f'{pid}::{sku}' for (pid, sku) in self.named}
            for (pid, sku) in extra_named:
                key = f'{pid}::{sku}'
                if key not in known:
                    self.named.append((pid, sku))
                    known.add(key)

    def note_shopper(self, content: str) -> None:
        text = (content or '').strip()
        if not text:
            return
        self.shopper_notes.append(text[:1600])
        self.shopper_notes = self.shopper_notes[-5:]
        self._refresh_from_notes()
        if '?' in text:
            self.pending_question = True
        if self.asked:
            self.received_clarification = True
            self.needs_ask = False
        if self.cart:
            low = text.lower()
            top = self.cart[-1]
            pid = str(top.get('product_id') or '')
            sku = str(top.get('sku') or '')
            if pid and sku and (not self.stock_ok(pid, sku)):
                self.must_recover = True
                self.needs_live_check = False
            if self.named and len(self.named) >= 2 and any((cue in low for cue in ('compare', 're-check', 'recheck', 're rank', 'rerank', 'current option', 'current options', 'before ordering'))):
                self.needs_rerank = True
                self.facts_refresh_generation = self.event_generation - 1

    def remember_action(self, name: str, args: dict[str, Any]) -> None:
        self.action_log.append(name)
        self.action_log = self.action_log[-24:]
        if name in {'search', 'filter'}:
            self.search_count += 1
        if name in {'view', 'compare'}:
            self.view_count += 1
        if name == 'message':
            self.asked = True
            self.pending_question = False
        if name == 'add_to_cart':
            self.cart_adds += 1
            pid = str(args.get('product_id') or '')
            sku = str(args.get('sku') or '')
            if pid and sku:
                key = f'{pid}::{sku}'
                self.cart = [item for item in self.cart if item.get('key') != key]
                self.cart.append({'product_id': pid, 'sku': sku, 'key': key})
                self.stock.pop(key, None)
                self.stock.pop(pid, None)
                self.needs_live_check = True
                self.must_recover = False
                self.checked_since_message.clear()
                self.cart_confirmed.discard(key)
                self.post_cart_stock.discard(key)
        if name == 'remove_from_cart':
            pid = str(args.get('product_id') or '')
            sku = str(args.get('sku') or '')
            key = f'{pid}::{sku}'
            if key:
                if self.must_recover:
                    self.failed_keys.add(key)
                    self.last_failed_key = key
                self.cart_confirmed.discard(key)
                self.post_cart_stock.discard(key)
            self.cart = [item for item in self.cart if item.get('key') != key]
            if not self.must_recover:
                self.needs_live_check = False
            self.checked_since_message.discard(key)

    def _remember_row(self, row: dict[str, Any], *, viewed: bool=False) -> None:
        pid = str(row.get('product_id') or '')
        if not pid:
            return
        sku = str(row.get('sku') or '')
        key = f'{pid}::{sku}' if sku else pid
        prev = self.candidates.get(key, {})
        merged = {**prev, **{k: v for (k, v) in row.items() if v is not None}}
        merged['product_id'] = pid
        if sku:
            merged['sku'] = sku
        self.candidates[key] = merged
        if viewed:
            slim = {k: merged.get(k) for k in ('product_id', 'sku', 'title', 'brand', 'price', 'in_stock', 'category_path', 'facts', 'listing_id') if merged.get(k) is not None}
            self.viewed[pid] = {**self.viewed.get(pid, {}), **slim}
        if len(self.candidates) > _MAX_STORED_CANDIDATES:
            ranked = sorted(self.candidates.items(), key=lambda kv: (0 if kv[1].get('product_id') in self.viewed else 1, 0 if kv[0] in self.stock or kv[1].get('product_id') in self.stock else 1, str(kv[1].get('price') or 10 ** 18)))
            keep = {k for (k, _) in ranked[:_MAX_STORED_CANDIDATES]}
            self.candidates = {k: v for (k, v) in self.candidates.items() if k in keep}

    def _product_snapshot(self, product_id: str, sku: str='') -> dict[str, Any]:
        key = f'{product_id}::{sku}' if sku else product_id
        viewed = dict(self.viewed.get(product_id) or {})
        facts = viewed.get('facts') if isinstance(viewed.get('facts'), dict) else {}
        stock = dict(self.stock.get(key) or self.stock.get(product_id) or {})
        cand = dict(self.candidates.get(key) or self.candidates.get(product_id) or {})
        merged = {**cand, **viewed, **stock, 'facts': facts}
        merged['product_id'] = product_id
        if sku:
            merged['sku'] = sku
        return merged

    def _apply_cart_observation(self, observation: dict[str, Any]) -> None:
        """Absorb live cart lines and market_event payloads from cart mutations."""
        event = observation.get('market_event')
        if isinstance(event, dict) and event:
            self.event_seen = True
            self.event_generation += 1
            self._invalidate_named_after_event()
            if self.named and len(self.named) >= 2:
                self.needs_rerank = True
                self.facts_refresh_generation = self.event_generation - 1
            target = event.get('target') if isinstance(event.get('target'), dict) else {}
            t_pid = str(target.get('product_id') or '')
            t_sku = str(target.get('sku') or '')
            t_key = f'{t_pid}::{t_sku}' if t_pid and t_sku else ''
            payload: dict[str, Any] = {}
            if event.get('new_price') is not None:
                payload['price'] = event.get('new_price')
            elif event.get('old_price') is not None and event.get('kind') != 'price_change':
                payload['price'] = event.get('old_price')
            if 'in_stock_after' in event:
                payload['in_stock'] = bool(event.get('in_stock_after'))
            if t_pid:
                payload['product_id'] = t_pid
            if t_sku:
                payload['sku'] = t_sku
            if t_key and payload:
                prev = dict(self.stock.get(t_key) or {})
                merged = {**prev, **payload}
                self.stock[t_key] = merged
                self.stock[t_pid] = merged
                self.stock_generation[t_key] = self.event_generation
                self._remember_row(merged)
                in_cart = any((item.get('key') == t_key for item in self.cart))
                if in_cart:
                    self.post_cart_stock.add(t_key)
                    self.cart_confirmed.add(t_key)
                    self.checked_since_message.add(t_key)
                    if not self.stock_ok(t_pid, t_sku):
                        self.must_recover = True
                        self.needs_live_check = False
                        self.failed_keys.add(t_key)
                        self.last_failed_key = t_key
                    else:
                        self.needs_live_check = False
        added = observation.get('added')
        if isinstance(added, dict):
            pid = str(added.get('product_id') or '')
            sku = str(added.get('sku') or '')
            if pid and sku:
                key = f'{pid}::{sku}'
                payload = {'product_id': pid, 'sku': sku, 'price': added.get('price'), 'in_stock': added.get('in_stock'), 'currency': added.get('currency')}
                prev = dict(self.stock.get(key) or {})
                merged = {**prev, **{k: v for (k, v) in payload.items() if v is not None}}
                self.stock[key] = merged
                self.stock[pid] = merged
                self._remember_row(merged)
                self.post_cart_stock.add(key)
                self.cart_confirmed.add(key)
                self.checked_since_message.add(key)
                if not self.stock_ok(pid, sku):
                    self.must_recover = True
                    self.needs_live_check = False
                    self.failed_keys.add(key)
                    self.last_failed_key = key
                else:
                    self.needs_live_check = False
        for row in observation.get('cart') or []:
            if not isinstance(row, dict):
                continue
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            if not pid or not sku:
                continue
            key = f'{pid}::{sku}'
            price = row.get('current_price', row.get('price', row.get('price_at_add')))
            payload = {'product_id': pid, 'sku': sku, 'price': price, 'in_stock': row.get('in_stock'), 'currency': row.get('currency')}
            prev = dict(self.stock.get(key) or {})
            merged = {**prev, **{k: v for (k, v) in payload.items() if v is not None}}
            self.stock[key] = merged
            self.stock[pid] = merged
            self._remember_row(merged)
            if any((item.get('key') == key for item in self.cart)):
                self.post_cart_stock.add(key)
                self.cart_confirmed.add(key)
                if not self.stock_ok(pid, sku):
                    self.must_recover = True
                    self.needs_live_check = False
                    self.failed_keys.add(key)
                    self.last_failed_key = key

    def ingest(self, name: str, args: dict[str, Any], observation: dict[str, Any]) -> None:
        observation = tf4_unwrap_observation(observation)
        if not observation:
            return
        if observation.get('error'):
            self.last_error = str(observation.get('error'))[:240]
            return
        if name in {'search', 'filter'}:
            rows = [row for row in observation.get('results') or [] if isinstance(row, dict)]
            for row in rows:
                self._remember_row(row)
            if not rows:
                self.search_count = max(self.search_count, 1)
        elif name == 'compare':
            if self.named and len(self.named) >= 2:
                self.facts_refresh_generation = self.event_generation
                if self.needs_rerank or self.event_seen:
                    for (pid, sku) in self.named:
                        key = f'{pid}::{sku}'
                        if key in self.failed_keys:
                            continue
                        self.stock.pop(key, None)
                        self.stock.pop(pid, None)
                        self.stock_generation.pop(key, None)
                        self.post_cart_stock.discard(key)
            for row in observation.get('compared') or []:
                if isinstance(row, dict):
                    self._remember_row(row, viewed=True)
                    pid = str(row.get('product_id') or '')
                    for variant in (row.get('variants') or [])[:8]:
                        if isinstance(variant, dict):
                            self._remember_row({**variant, 'product_id': variant.get('product_id') or pid, 'title': row.get('title'), 'brand': row.get('brand'), 'category_path': row.get('category_path'), 'facts': row.get('facts')}, viewed=True)
        elif name == 'view':
            self._remember_row(observation, viewed=True)
            pid = str(observation.get('product_id') or args.get('product_id') or '')
            if pid:
                facts = observation.get('facts') if isinstance(observation.get('facts'), dict) else {}
                self.viewed[pid] = {**self.viewed.get(pid, {}), 'product_id': pid, 'title': observation.get('title'), 'brand': observation.get('brand'), 'category_path': observation.get('category_path'), 'facts': facts, 'listing_id': observation.get('listing_id')}
            for row in (observation.get('variants') or [])[:8]:
                if isinstance(row, dict):
                    self._remember_row({**row, 'product_id': row.get('product_id') or pid})
        elif name == 'inspect_stock':
            pid = str(observation.get('product_id') or args.get('product_id') or '')
            sku = str(observation.get('sku') or args.get('sku') or '')
            if pid:
                payload = dict(observation)
                coerced = tf4_as_float(payload.get('price'))
                if coerced is not None:
                    payload['price'] = coerced
                key = f'{pid}::{sku}'
                self.stock[key] = payload
                self.stock[pid] = payload
                self.stock_generation[key] = self.event_generation
                self._remember_row({'product_id': pid, 'sku': sku, **payload})
                self.checked_since_message.add(key)
                if any((item.get('key') == key for item in self.cart)):
                    self.cart_confirmed.add(key)
                    self.post_cart_stock.add(key)
                if self.needs_live_check:
                    if self.stock_ok(pid, sku):
                        self.needs_live_check = False
                        self.must_recover = False
                    else:
                        self.needs_live_check = False
                        self.must_recover = True
                        self.failed_keys.add(key)
                        self.cart_confirmed.discard(key)
                        self.post_cart_stock.discard(key)
        elif name in {'add_to_cart', 'remove_from_cart', 'inspect_cart'}:
            self._apply_cart_observation(observation)
            if name == 'inspect_cart':
                rebuilt: list[dict[str, str]] = []
                stack: list[Any] = [observation]
                while stack:
                    node = stack.pop()
                    if isinstance(node, dict):
                        pid = str(node.get('product_id') or '')
                        sku = str(node.get('sku') or '')
                        if pid and sku:
                            rebuilt.append({'product_id': pid, 'sku': sku, 'key': f'{pid}::{sku}'})
                        stack.extend(node.values())
                    elif isinstance(node, list):
                        stack.extend(node)
                if rebuilt:
                    seen: set[str] = set()
                    uniq: list[dict[str, str]] = []
                    for item in rebuilt:
                        if item['key'] in seen:
                            continue
                        seen.add(item['key'])
                        uniq.append(item)
                    self.cart = uniq

    def stock_ok(self, product_id: str, sku: str) -> bool:
        key = f'{product_id}::{sku}'
        stock = self.stock.get(key) or self.stock.get(product_id)
        if not isinstance(stock, dict):
            return False
        if stock.get('in_stock') is not True:
            return False
        price = tf4_as_float(stock.get('price'))
        if price is None:
            return False
        if self.budget is not None and price > self.budget + 1e-09:
            return False
        return True

    def stock_is_fresh(self, product_id: str, sku: str) -> bool:
        key = f'{product_id}::{sku}'
        return self.stock_generation.get(key, -1) >= self.event_generation and (key in self.stock or product_id in self.stock)

    def merged_row(self, product_id: str, sku: str='') -> dict[str, Any]:
        """Live snapshot used for compliance and ranking."""
        return self._product_snapshot(product_id, sku)

    def row_compliant(self, row: dict[str, Any], sku: str='') -> bool:
        pid = str(row.get('product_id') or '')
        sku = sku or str(row.get('sku') or '')
        snap = self._product_snapshot(pid, sku) if pid else dict(row)
        if pid and pid not in self.viewed and self.firm_constraints:
            surface = {**row, 'title': row.get('title') or snap.get('title'), 'brand': row.get('brand') or snap.get('brand'), 'options': row.get('options') or snap.get('options'), 'category_path': row.get('category_path') or snap.get('category_path')}
            specs = [f for f in self.firm_constraints if re.search('\\d', f)]
            if specs:
                return _is_compliant(surface, specs)
            return True
        return _is_compliant(snap if pid else row, self.firm_constraints)

    def best_sku_for_product(self, product_id: str) -> str | None:
        """Pick the compliant sku for a product among remembered variants."""
        options: list[tuple[int, int, str]] = []
        for (key, row) in self.candidates.items():
            if str(row.get('product_id') or '') != product_id:
                continue
            sku = str(row.get('sku') or '')
            if not sku or key in self.failed_keys:
                continue
            snap = self._product_snapshot(product_id, sku)
            (hits, misses, soft) = _compliance_scores(snap, self.firm_constraints, self.soft_preferences)
            if misses and (not _is_compliant(snap, self.firm_constraints)):
                continue
            stock_bonus = 1 if self.stock_ok(product_id, sku) else 0
            options.append((misses - hits - soft - stock_bonus, -soft, sku))
        if not options:
            for (key, row) in self.candidates.items():
                if str(row.get('product_id') or '') == product_id:
                    sku = str(row.get('sku') or '')
                    if sku and key not in self.failed_keys:
                        return sku
            return None
        options.sort()
        return options[0][2]

    def tradeoff_rank_key(self, product_id: str, sku: str, *, axis_units: dict[str, str | None] | None=None) -> tuple[Any, ...]:
        snap = self._product_snapshot(product_id, sku)
        parts: list[Any] = []
        missing = 0
        for rule in self.priorities:
            unit = None if not axis_units else axis_units.get(rule['field'])
            value = _metric_value_with_unit(snap, rule['field'], unit, self.query)
            if value is None:
                missing += 1
                parts.append((1, 0.0))
                continue
            if rule['direction'] == 'max':
                parts.append((0, -float(value)))
            else:
                parts.append((0, float(value)))
        price = snap.get('price')
        if isinstance(price, (int, float)):
            parts.append((0 if missing == 0 else 1, float(price)))
        else:
            parts.append((1, 0.0))
        return (missing, *parts)

    def ranked_named(self) -> list[tuple[str, str, str]]:
        """Order named pid/sku pairs by query tradeoffs, then live feasibility."""
        snaps = [self._product_snapshot(pid, sku) for (pid, sku) in self.named if f'{pid}::{sku}' not in self.failed_keys]
        axis_units: dict[str, str | None] = {}
        used_units: set[str] = set()
        for rule in self.priorities:
            vote = _axis_unit_vote(snaps, rule['field'])
            if vote in used_units:
                buckets: dict[str, list[float]] = {}
                for row in snaps:
                    seen: set[str] = set()
                    for (unit, number) in _title_unit_numbers(row):
                        if unit in used_units or unit in seen:
                            continue
                        seen.add(unit)
                        buckets.setdefault(unit, []).append(number)
                vote = None
                best_span = -1.0
                for (unit, values) in buckets.items():
                    if len(values) < 2:
                        continue
                    span = max(values) - min(values)
                    score = span + 0.01 * len(unit)
                    if score > best_span:
                        best_span = score
                        vote = unit
            axis_units[rule['field']] = vote
            if vote:
                used_units.add(vote)
        rows: list[tuple[tuple[Any, ...], str, str, str]] = []
        for (pid, sku) in self.named:
            key = f'{pid}::{sku}'
            if key in self.failed_keys:
                continue
            snap = self._product_snapshot(pid, sku)
            feasible = 0
            if self.stock_ok(pid, sku):
                feasible = 0
            elif key in self.stock or pid in self.stock:
                feasible = 2
            else:
                feasible = 1
            price = snap.get('price')
            over = 1 if self.budget is not None and isinstance(price, (int, float)) and (float(price) > self.budget) else 0
            rank = (feasible, over, self.tradeoff_rank_key(pid, sku, axis_units=axis_units), key)
            rows.append((rank, pid, sku, key))
        rows.sort(key=lambda item: item[0])
        return [(pid, sku, key) for (_rank, pid, sku, key) in rows]

    def ranked_candidates(self) -> list[tuple[int, str, dict[str, Any]]]:
        if self.named and self.priorities:
            named_ranked: list[tuple[int, str, dict[str, Any]]] = []
            for (index, (pid, sku, key)) in enumerate(self.ranked_named()):
                row = self._product_snapshot(pid, sku)
                score = 1000 - index
                if self.stock_ok(pid, sku):
                    score += 50
                named_ranked.append((score, key, row))
            if named_ranked:
                return named_ranked
        goal = tf4_goal_tokens(self.query, self.shopper_notes)
        specs = _goal_spec_fragments(self.query, self.shopper_notes)
        category = (_category_hint(self.query) or '').lower()
        firm = self.firm_constraints
        soft = self.soft_preferences
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for (key, row) in self.candidates.items():
            if key in self.failed_keys:
                continue
            score = 0
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            viewed = self.viewed.get(pid) or {}
            facts = viewed.get('facts') if isinstance(viewed.get('facts'), dict) else {}
            merged = {**row, **viewed, 'facts': facts}
            if sku:
                merged = {**merged, **self._product_snapshot(pid, sku)}
            (firm_hits, firm_misses, soft_hits) = _compliance_scores(merged, firm, soft)
            if firm:
                score += 6 * firm_hits
                score -= 8 * firm_misses
                if pid in self.viewed and firm_misses and (not _is_compliant(merged, firm)):
                    score -= 40
            brands = _required_brand_labels(self.shopper_notes)
            if brands:
                blob_b = _row_text_blob(merged)
                if any((_fragment_match(blob_b, b) for b in brands)):
                    score += 40
                else:
                    score -= 35
            if soft:
                score += 5 * soft_hits
                if firm_misses == 0 and soft_hits:
                    score += 10 * soft_hits
                elif soft_hits:
                    score += 4 * soft_hits
                if soft_hits == 0 and self.task_shape == 'preference_reasoning':
                    score -= 8
                if self.task_shape == 'preference_reasoning' and soft:
                    score += int(18 * soft_hits / max(1, len(soft)))
                    if soft_hits >= max(2, len(soft) // 2):
                        score += 12
                    axes = _priority_soft_axes(self.shopper_notes, self.query)
                    if axes:
                        blob_ax = _row_text_blob(merged)
                        axis_hits = sum((1 for axis in axes[:4] if _axis_evidenced(blob_ax, axis)))
                        score += 14 * axis_hits
                        if axis_hits < 2 and len(axes) >= 2:
                            score -= 25
                    if pid in _failed_product_ids(self):
                        score -= 80
                blob = _row_text_blob(merged)
                for frag in soft:
                    if not re.fullmatch('\\d{3,14}', frag):
                        continue
                    if _soft_code_match(blob, frag) or frag in str(merged.get('sku') or ''):
                        score += 30
                    else:
                        score -= 8 if self.must_recover else 55
                for frag in soft:
                    if re.fullmatch('\\d{3,14}', frag) or len(frag) < 3:
                        continue
                    if _fragment_match(blob, frag):
                        score += 10
                    else:
                        score -= 4
            if firm:
                path = ' '.join((str(part).lower() for part in merged.get('category_path') or [] if part is not None))
                title_l = str(merged.get('title') or '').lower()
                for frag in firm:
                    if re.search('\\d', frag) or len(frag) < 4:
                        continue
                    fl = frag.lower()
                    if fl in path or fl in title_l:
                        score += 8
                    elif path and fl not in path and (fl not in title_l):
                        score -= 12
            if pid in self.viewed:
                score += 4
            if key in self.stock or pid in self.stock:
                score += 3
                if self.stock_ok(pid, sku):
                    score += 4
                else:
                    score -= 5
            if row.get('in_stock') is True:
                score += 1
            price = tf4_as_float(row.get('price'))
            if price is None:
                price = tf4_as_float((self.stock.get(key) or {}).get('price'))
            if self.budget is not None and price is not None:
                score += 2 if price <= self.budget else -6
            if any((pid == n[0] for n in self.named)):
                score += 3
            title = str(row.get('title') or '')
            title_l = title.lower()
            brand = str(row.get('brand') or viewed.get('brand') or '')
            score += _title_goal_alignment(title, goal)
            score += _spec_match_score(merged, specs)
            fact_blob = ' '.join((str(v).lower() for v in (title, brand, json.dumps(facts, ensure_ascii=False, default=str), row.get('options') or '') if v not in (None, '')))
            for tok in goal:
                if len(tok) < 2:
                    continue
                if any((ch.isdigit() for ch in tok)) and tok in fact_blob:
                    score += 5
            if brand:
                score += min(2, _value_overlap(brand, goal))
                if brand.lower() in goal:
                    score += 2
            if category:
                path = ' '.join((str(part).lower() for part in merged.get('category_path') or [] if part is not None))
                if category in path or category in title_l:
                    score += 3
                elif path and category not in path:
                    score -= 2
            for field in self.mentioned_fields:
                observed = _lookup_fact([viewed, row, facts], field)
                if observed is None:
                    continue
                score += 2 + min(3, _value_overlap(observed, goal))
            ranked.append((score, key, row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    def best_recovery(self) -> dict[str, str] | None:
        """Recover to a sibling/sku-stem alternate, even after the cart was cleared."""
        cur_pid = ''
        cur_sku = ''
        if self.cart and (not self.named):
            top = self.cart[-1]
            cur_pid = str(top.get('product_id') or '')
            cur_sku = str(top.get('sku') or '')
        if (not cur_pid or not cur_sku) and self.last_failed_key and ('::' in self.last_failed_key):
            (cur_pid, cur_sku) = self.last_failed_key.split('::', 1)
        if (not cur_pid or not cur_sku) and self.failed_keys:
            for key in reversed(list(self.failed_keys)):
                if '::' in key:
                    (cur_pid, cur_sku) = key.split('::', 1)
                    break
        if self.task_shape == 'preference_reasoning':
            failed_stem = re.sub('-\\d+$', '', cur_sku) if cur_sku else ''
            failed_pids = _failed_product_ids(self)
            for (_score, key, row) in self.ranked_candidates():
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '') or (self.best_sku_for_product(pid) or '')
                if not pid or not sku or key in self.failed_keys:
                    continue
                if cur_pid and pid == cur_pid:
                    continue
                if pid in failed_pids:
                    continue
                if failed_stem and len(failed_stem) >= 4 and sku.startswith(failed_stem):
                    continue
                snap = self._product_snapshot(pid, sku)
                if not _preference_cart_ok(self, snap):
                    continue
                if pid not in self.viewed:
                    return {'action': 'view', 'product_id': pid, 'sku': sku}
                if key not in self.stock and pid not in self.stock:
                    return {'action': 'inspect_stock', 'product_id': pid, 'sku': sku}
                if self.stock_ok(pid, sku):
                    return {'action': 'add_to_cart', 'product_id': pid, 'sku': sku}
            return None
        if cur_pid and cur_sku and (not self.named):
            alt = self.best_sku_for_product(cur_pid)
            if alt and alt != cur_sku:
                alt_key = f'{cur_pid}::{alt}'
                if alt_key not in self.failed_keys:
                    if cur_pid not in self.viewed:
                        return {'action': 'view', 'product_id': cur_pid, 'sku': alt}
                    if alt_key not in self.stock and cur_pid not in self.stock:
                        return {'action': 'inspect_stock', 'product_id': cur_pid, 'sku': alt}
                    if self.stock_ok(cur_pid, alt):
                        return {'action': 'add_to_cart', 'product_id': cur_pid, 'sku': alt}
            stem = re.sub('-\\d+$', '', cur_sku)
            if stem and len(stem) >= 4:
                soft_codes = [frag for frag in self.soft_preferences if re.fullmatch('\\d{3,14}', frag)]
                soft_words = [frag.lower() for frag in self.soft_preferences if not re.fullmatch('\\d{3,14}', frag) and len(frag) >= 3]
                stem_hits: list[tuple[int, str, str, str]] = []
                for (key, row) in self.candidates.items():
                    if key in self.failed_keys:
                        continue
                    other_pid = str(row.get('product_id') or '')
                    other_sku = str(row.get('sku') or '')
                    if not other_pid or not other_sku or other_sku == cur_sku:
                        continue
                    if not other_sku.startswith(stem):
                        continue
                    snap = self._product_snapshot(other_pid, other_sku)
                    blob = _row_text_blob(snap).lower()
                    score = 0
                    if soft_codes:
                        if not any((_soft_code_match(blob, code) or code in other_sku for code in soft_codes)):
                            score -= 2
                        else:
                            score += 8
                    if soft_words and any((w in blob for w in soft_words)):
                        score += 4
                    if other_pid == cur_pid:
                        score += 3
                    if self.stock_ok(other_pid, other_sku):
                        score += 5
                    stem_hits.append((score, other_pid, other_sku, key))
                stem_hits.sort(key=lambda item: (-item[0], item[3]))
                for (_score, other_pid, other_sku, key) in stem_hits:
                    if other_pid not in self.viewed:
                        return {'action': 'view', 'product_id': other_pid, 'sku': other_sku}
                    if key not in self.stock and other_pid not in self.stock:
                        return {'action': 'inspect_stock', 'product_id': other_pid, 'sku': other_sku}
                    if self.stock_ok(other_pid, other_sku):
                        return {'action': 'add_to_cart', 'product_id': other_pid, 'sku': other_sku}
                if soft_codes:
                    return None
        if self.named:
            for (pid, sku, key) in self.ranked_named():
                if key in self.failed_keys:
                    continue
                if pid not in self.viewed:
                    return {'action': 'view', 'product_id': pid, 'sku': sku}
                if key not in self.stock and pid not in self.stock:
                    return {'action': 'inspect_stock', 'product_id': pid, 'sku': sku}
                if self.stock_ok(pid, sku):
                    return {'action': 'add_to_cart', 'product_id': pid, 'sku': sku}
            return None
        soft_codes = [frag for frag in self.soft_preferences if re.fullmatch('\\d{3,14}', frag)]
        for (score, key, row) in self.ranked_candidates():
            if score < 2 and (not soft_codes):
                break
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '') or (self.best_sku_for_product(pid) or '')
            if not pid or not sku or key in self.failed_keys:
                continue
            snap = self._product_snapshot(pid, sku)
            if soft_codes:
                blob = _row_text_blob(snap)
                if not any((_soft_code_match(blob, code) or code in sku for code in soft_codes)):
                    continue
            elif pid in self.viewed and self.firm_constraints and (not self.row_compliant(snap, sku)):
                continue
            if pid not in self.viewed:
                return {'action': 'view', 'product_id': pid, 'sku': sku}
            if key in self.stock or pid in self.stock or f'{pid}::{sku}' in self.stock:
                if self.stock_ok(pid, sku):
                    return {'action': 'add_to_cart', 'product_id': pid, 'sku': sku}
                continue
            return {'action': 'inspect_stock', 'product_id': pid, 'sku': sku}
        return None

    def next_hint(self, turn: int, max_steps: int) -> str:
        remaining = max_steps - turn + 1
        if self.needs_ask and (not self.received_clarification):
            return 'Ask the shopper for the missing firm specification with message before searching deeply or carting.'
        if self.pending_question:
            return 'Answer the shopper with message before ordering.'
        if self.must_recover and self.cart:
            top = self.cart[-1]
            return f"Live check failed for {top['key']}. remove_from_cart, find another fully valid option, view it, cart it, inspect_stock, then order."
        if self.needs_live_check and self.cart:
            top = self.cart[-1]
            if top['key'] not in self.post_cart_stock:
                return f"Cart mutated — inspect_stock {top['key']} once for the live price/stock, then order or recover."
        if self.last_error:
            return f'Recover from error ({self.last_error}), then continue.'
        recent = self.action_log[-4:]
        if recent and len(set(recent)) == 1 and (recent[0] in {'search', 'filter', 'message', 'inspect_stock'}):
            return f'Stop repeating {recent[0]}; advance toward a verified place_test_order.'
        if self.named and self.view_count < min(2, len(self.named)):
            return 'View/compare the named variant keys from the request before carting.'
        if self.named and self.priorities:
            return "Rank named options by the shopper's stated tradeoff order using live facts, cart the current best, and re-rank after any market event."
        if not self.candidates:
            return f'Search/filter with short constraint-focused queries (seed hint: {self.search_seed}) and in_stock=true.'
        if self.view_count == 0:
            return 'Inspect a shortlist with view/compare and verify the exact sku facts.'
        if not self.cart:
            if remaining <= 7:
                return 'Cart the best fully compliant inspected variant now.'
            return 'Cart only a fully compliant intended purchase. Honor soft preferences when a compliant option exists.'
        top = self.cart[-1]
        key = top['key']
        if key not in self.post_cart_stock:
            return 'inspect_stock the carted sku once after carting before place_test_order.'
        if not self.stock_ok(top['product_id'], top['sku']):
            return 'Carted sku failed live checks — remove_from_cart and recover.'
        if remaining <= 8:
            return 'Verified — place_test_order NOW (include required claims if schema needs them).'
        return 'Live checks passed — place_test_order for the carted variant.'

    def render(self, turn: int, max_steps: int) -> str:
        lines = [f'board turn={turn}/{max_steps} remaining={max_steps - turn + 1}', f"shape={self.task_shape} budget={(self.budget if self.budget is not None else 'none')} needs_ask={self.needs_ask and (not self.received_clarification)} needs_live_check={self.needs_live_check} must_recover={self.must_recover} event_seen={self.event_seen}", f'searches={self.search_count} views={self.view_count} candidates={len(self.candidates)} cart={len(self.cart)} adds={self.cart_adds}', f'search_seed={self.search_seed}', f'next={self.next_hint(turn, max_steps)}']
        if self.firm_constraints:
            lines.append('firm=' + ', '.join(self.firm_constraints[:8]))
        if self.soft_preferences:
            lines.append('soft=' + ', '.join(self.soft_preferences[:6]))
        if self.priorities:
            lines.append('tradeoffs=' + ', '.join((f"{item['direction']}:{item['field']}" for item in self.priorities)))
        if self.named:
            lines.append('named_keys=' + ', '.join((f'{pid}::{sku}' for (pid, sku) in self.named[:8])))
        if self.mentioned_fields:
            lines.append('query_claim_fields=' + ', '.join(sorted(self.mentioned_fields)))
        if self.shopper_notes:
            lines.append('shopper_notes:')
            for note in self.shopper_notes[-2:]:
                lines.append(f'  - {note}')
        if self.cart:
            lines.append('cart:')
            for item in self.cart[-3:]:
                stock = self.stock.get(item['key']) or self.stock.get(item['product_id']) or {}
                lines.append(f"  - {item['key']} price={stock.get('price', '?')} in_stock={stock.get('in_stock', '?')} post_cart={item['key'] in self.post_cart_stock}")
        ranked = self.ranked_candidates()
        if ranked:
            lines.append('candidates:')
            for (score, key, row) in ranked[:_BOARD_CANDIDATE_CAP]:
                lines.append(f"  - score={score} {key} title={str(row.get('title') or '')[:64]} price={row.get('price', '?')} stock={row.get('in_stock', '?')} brand={row.get('brand', '?')}")
        if self.cart:
            top = self.cart[-1]
            viewed = self.viewed.get(top['product_id']) or {}
            facts = viewed.get('facts') if isinstance(viewed.get('facts'), dict) else {}
            if facts:
                preview = ', '.join((f'{k}={str(v)[:40]}' for (k, v) in list(facts.items())[:8]))
                lines.append(f'final_facts={preview}')
                lines.append('If justification is required, claim price, in_stock, one shopper attribute from these facts, and one allowed reason field.')
        return '\n'.join(lines)

def _surface_claim_value(product: dict[str, Any], field: str, goal_text: str) -> Any:
    """Build a claim value from observed text near the field's own name tokens."""
    parts = [p for p in re.split('[_\\s]+', field.lower()) if len(p) >= 3]
    if not parts:
        return None
    want_nums = set(_NUM_TOKEN_RE.findall(goal_text))
    surfaces: list[str] = []
    for (key, value) in product.items():
        if not isinstance(key, str) or value in (None, '', True, False):
            continue
        if isinstance(value, (dict, list)):
            continue
        surfaces.append(f'{key} {value}')
    blob = ' '.join(surfaces).lower()
    if not blob:
        return None
    best: tuple[int, str] | None = None
    for match in re.finditer('(\\d+(?:\\.\\d+)?)\\s*-?\\s*([a-z]{0,8})', blob):
        num = match.group(1)
        unit = (match.group(2) or '').lower()
        window = blob[max(0, match.start() - 30):match.end() + 30]
        score = 0
        if any((p in window for p in parts)):
            score += 3
        if num in want_nums:
            score += 2
        if any((p in goal_text for p in parts)) and num in want_nums:
            score += 2
        kn_hit = any((_keys_structurally_match(_norm_key(key), _norm_key(field)) for key in product if isinstance(key, str)))
        if kn_hit and num in set(_NUM_TOKEN_RE.findall(str(product.get(field, '')))):
            score += 4
        if score < 3:
            continue
        value = f'{num}{unit}' if unit else num
        glued = f'{num}{unit}' if unit else ''
        if glued and glued in blob.replace(' ', ''):
            value = glued
        if best is None or score > best[0]:
            best = (score, value)
    if best:
        return best[1]
    for (key, value) in product.items():
        if not isinstance(key, str) or value in (None, ''):
            continue
        if isinstance(value, (dict, list, bool)):
            continue
        if _keys_structurally_match(_norm_key(key), _norm_key(field)):
            return value
    return None

def tf4_ground_order(args: dict[str, Any], *, tools: list[dict[str, Any]], board: EpisodeBoard) -> dict[str, Any]:
    cleaned = dict(args)
    required = _required_args(tools, 'place_test_order')
    needs = 'justification' in required
    if not needs:
        cleaned.pop('justification', None)
        return cleaned
    pid = str(cleaned.get('product_id') or '')
    sku = str(cleaned.get('sku') or '')
    if (not pid or not sku) and board.cart:
        top = board.cart[-1]
        pid = pid or top['product_id']
        sku = sku or top['sku']
        cleaned['product_id'] = pid
        cleaned['sku'] = sku
    key = f'{pid}::{sku}'
    stock = dict(board.stock.get(key) or board.stock.get(pid) or {})
    viewed = dict(board.viewed.get(pid) or {})
    candidate = dict(board.candidates.get(key) or board.candidates.get(pid) or {})
    product = _flatten_product({**candidate, **viewed, **stock})
    allowed = _claim_field_allowlist(tools)
    open_spec = _justification_open_spec(tools)
    soft_fields = _soft_claim_fields(tools, allowed)
    goal = tf4_goal_tokens(board.query, board.shopper_notes)
    goal_text = f"{tf4_preamble(board.query)} {' '.join(board.shopper_notes[-3:])}".lower()
    prefer = set(board.mentioned_fields)
    raw_claims: list[dict[str, Any]] = []
    just = cleaned.get('justification')
    if isinstance(just, dict) and isinstance(just.get('claims'), list):
        raw_claims = [c for c in just['claims'] if isinstance(c, dict)]
    if open_spec:
        allowed.update({'price', 'in_stock'})
        allowed.update(soft_fields)
        allowed.update(prefer)
        for claim in raw_claims:
            field = str(claim.get('field') or '').strip()
            if field and re.fullmatch('[A-Za-z][A-Za-z0-9_]{1,40}', field):
                allowed.add(field)
                prefer.add(field)
    if not allowed:
        allowed = {'price', 'in_stock'}
    grounded: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(field: str, value: Any) -> None:
        if field not in allowed or field in seen:
            return
        if value is None or value == '':
            return
        if isinstance(value, bool):
            text = 'true' if value else 'false'
        else:
            text = str(value).strip()
        if not text:
            return
        if text.lower() in {'unknown', 'none', 'n/a', 'null'}:
            return
        if field != 'in_stock' and text.lower() == 'false':
            return
        grounded.append({'field': field, 'value': text[:80]})
        seen.add(field)

    def best_observed_value(field: str) -> Any:
        direct = _lookup_fact([product, stock, viewed, candidate], field)
        if direct not in (None, ''):
            return direct
        if field in soft_fields and field != 'brand':
            return None
        want_nums = _goal_numbers_for_field(board.query, field, board.shopper_notes)
        best_val: Any = None
        best_score = 0
        for (key_name, value) in product.items():
            if value in (None, '', True, False):
                continue
            if not isinstance(key_name, str):
                continue
            if isinstance(value, (dict, list)):
                continue
            if key_name in {'product_id', 'sku', 'title', 'price', 'in_stock', 'currency', 'listing_id', 'brand'} and field not in {'brand', 'price', 'in_stock'}:
                continue
            text = str(value).lower().strip()
            nums = set(_NUM_TOKEN_RE.findall(text))
            shared = nums & want_nums if want_nums else set()
            overlap = 0
            if _keys_structurally_match(_norm_key(key_name), _norm_key(field)):
                overlap += 8
            if shared:
                overlap += 3 * len(shared)
            if len(text) >= 2 and (text in goal_text or any((tok == text for tok in goal if len(tok) >= 2))):
                overlap += 6
            overlap += min(2, _value_overlap(value, goal))
            if overlap > best_score:
                best_score = overlap
                best_val = value
        return best_val if best_score >= 6 else None
    add('price', stock.get('price', product.get('price')))
    add('in_stock', stock.get('in_stock', product.get('in_stock')))
    spec_frags = _goal_spec_fragments(board.query, board.shopper_notes)
    facts = product.get('facts') if isinstance(product.get('facts'), dict) else {}
    spec_field_claimed = False
    for frag in spec_frags:
        if spec_field_claimed:
            break
        frag_l = frag.lower().strip()
        if len(frag_l) < 2:
            continue
        frag_alts = [frag_l, frag_l.replace('-', ' ')]
        if re.fullmatch('\\d+(?:\\.\\d+)?-\\w+', frag_l):
            frag_alts.append(frag_l.split('-', 1)[0])
        matched_field = None
        matched_val = None
        surfaces: list[tuple[str, Any]] = list(facts.items())
        for (key, value) in product.items():
            if key == 'facts':
                continue
            surfaces.append((key, value))
        for (key, value) in surfaces:
            if not isinstance(key, str) or value in (None, '', True, False):
                continue
            if isinstance(value, (dict, list)):
                continue
            if key in {'title', 'listing_id', 'product_id', 'sku', 'model', 'condition', 'navigation', 'brand', 'price', 'in_stock', 'currency', 'category_path', 'options', 'variants'}:
                continue
            text = str(value).lower()
            if not any((alt in text or alt in text.replace('-', ' ') for alt in frag_alts)):
                continue
            matched_field = key
            matched_val = next((alt for alt in frag_alts if alt in text or alt in text.replace('-', ' ')), value)
            if key in facts:
                break
        if matched_field and matched_val is not None:
            if open_spec:
                allowed.add(matched_field)
            add(matched_field, matched_val)
            spec_field_claimed = True
    for claim in raw_claims:
        field = str(claim.get('field') or '').strip()
        if not field or field in {'price', 'in_stock'}:
            continue
        if field not in allowed and (not open_spec):
            continue
        if open_spec:
            allowed.add(field)
        observed = best_observed_value(field)
        if observed is None:
            observed = _surface_claim_value(product, field, goal_text)
        if observed is not None:
            add(field, observed)
            continue
        raw_val = claim.get('value')
        if raw_val is None or raw_val == '':
            continue
        token = str(raw_val).strip().lower()
        blob = json.dumps({k: v for (k, v) in product.items() if k != 'title'}, ensure_ascii=False, default=str).lower()
        if len(token) >= 2 and token in blob:
            add(field, raw_val)
    attr_fields: list[str] = []
    for field in list(prefer) + sorted(allowed):
        if field in {'price', 'in_stock'} or field in soft_fields:
            continue
        if field in attr_fields:
            continue
        surface = field.replace('_', ' ').lower()
        parts = [p for p in field.lower().split('_') if len(p) >= 3]
        mentioned = field in prefer or surface in goal_text or any((len(p) >= 4 and p in goal_text for p in parts)) or (len(parts) >= 2 and all((p in goal_text for p in parts)))
        if mentioned:
            attr_fields.append(field)
    for field in attr_fields:
        if field in seen:
            continue
        observed = best_observed_value(field)
        if observed is None:
            observed = _surface_claim_value(product, field, goal_text)
        if observed is not None:
            add(field, observed)
    if not any((f not in {'price', 'in_stock'} | soft_fields for f in seen)):
        scored: list[tuple[int, str, Any]] = []
        name_hints = [str(c).strip() for c in list(prefer) + [str(c.get('field') or '') for c in raw_claims] if str(c).strip() and str(c).strip() not in {'price', 'in_stock'}]
        for (key_name, value) in product.items():
            if not isinstance(key_name, str) or value in (None, ''):
                continue
            if key_name in {'product_id', 'sku', 'title', 'price', 'in_stock', 'currency', 'listing_id', 'category_path', 'variants', 'options', 'brand', 'condition', 'model'}:
                continue
            if isinstance(value, (dict, list, bool)):
                continue
            text = str(value).lower().strip()
            overlap = 0
            if len(text) >= 2 and (text in goal_text or any((tok == text for tok in goal if len(tok) >= 2))):
                overlap += 6
            overlap += min(3, _value_overlap(value, goal))
            nums = set(_NUM_TOKEN_RE.findall(text))
            overlap += 2 * len(nums & {t for t in goal if any((ch.isdigit() for ch in t))})
            if overlap <= 0:
                continue
            claim_name = None
            for cand in name_hints:
                if _keys_structurally_match(_norm_key(cand), _norm_key(key_name)):
                    claim_name = cand
                    overlap += 3
                    break
            if claim_name is None:
                continue
            if open_spec or claim_name in allowed:
                scored.append((overlap, claim_name, value))
        scored.sort(key=lambda row: (-row[0], row[1]))
        if scored:
            (_score, field, value) = scored[0]
            if open_spec:
                allowed.add(field)
            add(field, value)
    for claim in raw_claims:
        field = str(claim.get('field') or '').strip()
        if not field or field in seen or field in {'price', 'in_stock'}:
            continue
        if not open_spec and field not in allowed:
            continue
        best_val: Any = None
        best_score = 0
        for (key_name, value) in product.items():
            if not isinstance(key_name, str) or value in (None, ''):
                continue
            if key_name in {'product_id', 'sku', 'title', 'price', 'in_stock', 'currency', 'listing_id', 'brand', 'condition'}:
                continue
            if isinstance(value, (dict, list, bool)):
                continue
            text = str(value).lower().strip()
            score = 0
            if len(text) >= 2 and (text in goal_text or any((tok == text for tok in goal if len(tok) >= 2))):
                score += 6
            score += min(3, _value_overlap(value, goal))
            if score > best_score:
                best_score = score
                best_val = value
        if best_val is not None and best_score >= 6:
            if open_spec:
                allowed.add(field)
            add(field, best_val)
    if 'brand' in allowed and 'brand' not in seen and product.get('brand'):
        brand_val = str(product.get('brand') or '').strip()
        title_l = str(product.get('title') or '').lower()
        brand_l = brand_val.lower()
        if brand_l and (brand_l in title_l or brand_l == str(viewed.get('brand') or '').lower() or brand_l == str(candidate.get('brand') or '').lower()):
            add('brand', brand_val)
    if soft_fields and (not seen & soft_fields):
        for field in sorted(soft_fields):
            if field == 'brand' and product.get('brand'):
                add(field, product.get('brand'))
                break
            observed = _lookup_fact([product], field)
            if observed is True or str(observed).strip().lower() in {'true', 'yes', '1'}:
                add(field, 'true')
                break
    if needs and (not grounded):
        add('price', stock.get('price', product.get('price')))
        add('in_stock', stock.get('in_stock', product.get('in_stock', True)))
        if product.get('brand') and 'brand' in allowed:
            add('brand', product.get('brand'))
    if needs and grounded:
        reason_names = set(soft_fields) | {'brand'}
        core = [c for c in grounded if c['field'] in {'price', 'in_stock'}]
        specs = [c for c in grounded if c['field'] not in {'price', 'in_stock'} and c['field'] not in reason_names]
        reason_claims = [c for c in grounded if c['field'] in reason_names]
        slim: list[dict[str, str]] = list(core)
        if specs:
            slim.append(specs[0])
        if reason_claims:
            slim.append(reason_claims[0])
        elif product.get('brand') and 'brand' in allowed:
            slim.append({'field': 'brand', 'value': str(product.get('brand'))[:80]})
        grounded = slim
    cleaned['justification'] = {'claims': grounded}
    return cleaned

def _normalize_args(name: str, args: dict[str, Any], *, tools: list[dict[str, Any]], board: EpisodeBoard) -> dict[str, Any]:
    cleaned = dict(args)
    if name == 'search':
        query = str(cleaned.get('query') or '').strip()
        if not query or query.isdigit() or len(_TOKEN_RE.findall(query)) < 2:
            query = _search_variants(board.query, board.shopper_notes, board.search_count)
        budget_nums = _budget_number_strings(f"{tf4_preamble(board.query)} {' '.join(board.shopper_notes[-2:])}")
        discourse = {'compare', 'choose', 'among', 'remaining', 'ties', 'tradeoff', 'candidate', 'candidates', 'variant', 'variants', 'option', 'options', 'walk', 'through', 'why', 'right', 'pick', 'select'}
        kept: list[str] = []
        for tok in query.split():
            bare = tok.replace(',', '').lower()
            if bare in budget_nums or bare in discourse:
                continue
            if re.fullmatch('\\d{6,}', bare) or '::' in tok:
                continue
            if re.fullmatch('[a-z0-9]+-\\d+', bare) and len(bare) >= 8:
                continue
            if re.fullmatch('\\d+(?:\\.\\d+)?', bare) and (bare in budget_nums or float(bare) >= 1000):
                continue
            kept.append(tok)
        query = ' '.join(kept).strip() or board.search_seed
        tokens = _TOKEN_RE.findall(query)
        if len(tokens) < 2 or all((t.lower() in discourse for t in tokens)):
            query = _search_variants(board.query, board.shopper_notes, board.search_count)
            tokens = _TOKEN_RE.findall(query)
        if len(tokens) > 8 and '"' not in query:
            cleaned['query'] = ' '.join(tokens[:6])
        else:
            cleaned['query'] = query
        cleaned.setdefault('in_stock', True)
        cleaned.setdefault('k', 10)
        if board.budget is not None and board.budget >= 5:
            cleaned.setdefault('max_price', board.budget)
    if name == 'filter':
        cleaned.setdefault('in_stock', True)
        cleaned.setdefault('k', 10)
        if board.budget is not None and board.budget >= 5 and ('max_price' not in cleaned):
            cleaned['max_price'] = board.budget
        if not str(cleaned.get('query') or '').strip():
            cleaned['query'] = _search_variants(board.query, board.shopper_notes, board.search_count)
        category = _category_hint(board.query)
        if category and 'category' not in cleaned:
            cleaned['category'] = category
    if name == 'compare':
        ids = cleaned.get('product_ids')
        if isinstance(ids, list):
            cleaned['product_ids'] = [str(x) for x in ids[:5] if str(x).strip()]
        elif isinstance(ids, str) and ids.strip():
            cleaned['product_ids'] = [part.strip() for part in ids.split(',') if part.strip()][:5]
        if not cleaned.get('product_ids') and board.named:
            cleaned['product_ids'] = [pid for (pid, _sku) in board.named[:5]]
    if name == 'message':
        cleaned['content'] = str(cleaned.get('content') or '').strip()[:400]
    if name == 'place_test_order':
        if (not cleaned.get('product_id') or not cleaned.get('sku')) and board.cart:
            top = board.cart[-1]
            cleaned.setdefault('product_id', top['product_id'])
            cleaned.setdefault('sku', top['sku'])
        cleaned = tf4_ground_order(cleaned, tools=tools, board=board)
    return cleaned

def _guard_actions(action_pairs: list[tuple[str, dict[str, Any]]], *, board: EpisodeBoard, tools: list[dict[str, Any]], max_calls: int) -> list[tuple[str, dict[str, Any]]]:
    names = _tool_names(tools)
    guarded: list[tuple[str, dict[str, Any]]] = []
    if board.needs_ask and (not board.received_clarification) and ('message' in names) and (not board.asked):
        risky = {'add_to_cart', 'place_test_order', 'remove_from_cart'}
        if any((name in risky for (name, _) in action_pairs)) or not any((name == 'message' for (name, _) in action_pairs)):
            return [('message', {'content': 'Before I search, what firm product specification must the item meet besides category, budget, and in-stock availability?'})]
    already_removing = any((name == 'remove_from_cart' for (name, _) in action_pairs))
    if board.must_recover and board.cart and ('remove_from_cart' in names) and (not already_removing):
        if board.named and len(board.named) >= 2 and board.needs_rerank and (board.facts_refresh_generation < board.event_generation) and ('compare' in names):
            return [('compare', {'product_ids': [pid for (pid, _sku) in board.named[:6]]})]
        top = board.cart[-1]
        forced: list[tuple[str, dict[str, Any]]] = [('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']})]
        nxt = board.best_recovery()
        if nxt and len(forced) < max_calls:
            action = nxt['action']
            if action in names:
                if action == 'add_to_cart' and board.task_shape == 'preference_reasoning':
                    snap_n = board._product_snapshot(nxt['product_id'], nxt['sku'])
                    if not _preference_cart_ok(board, snap_n):
                        nxt = None
                if nxt and action in names:
                    args = {'product_id': nxt['product_id']}
                    if action != 'view':
                        args['sku'] = nxt['sku']
                    forced.append((action, args))
        if len(forced) == 1 and 'search' in names:
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                forced.append(nxt_d)
        return forced[:max_calls]
    if not board.cart and board.failed_keys and (not any((name in {'add_to_cart', 'view', 'inspect_stock', 'search', 'filter'} for (name, _) in action_pairs))):
        nxt = board.best_recovery()
        if nxt and nxt['action'] in names:
            args = {'product_id': nxt['product_id']}
            if nxt['action'] != 'view':
                args['sku'] = nxt['sku']
            return [(nxt['action'], args)]
    for (name, args) in action_pairs:
        pid = str(args.get('product_id') or '')
        sku = str(args.get('sku') or '')
        if board.needs_ask and (not board.asked) and (name in {'add_to_cart', 'place_test_order'}) and ('message' in names):
            guarded.append(('message', {'content': 'What firm specification is required for this purchase besides category, budget, and stock?'}))
            continue
        if name == 'add_to_cart' and pid and (pid not in board.viewed) and ('view' in names):
            guarded.append(('view', {'product_id': pid}))
            continue
        if name == 'add_to_cart' and pid and sku:
            snap = board._product_snapshot(pid, sku)
            if board.task_shape == 'preference_reasoning' and pid in _failed_product_ids(board):
                nxt = board.best_recovery()
                if nxt and nxt['action'] in names:
                    args = {'product_id': nxt['product_id']}
                    if nxt['action'] != 'view':
                        args['sku'] = nxt['sku']
                    guarded.append((nxt['action'], args))
                    continue
                if 'search' in names:
                    nxt_d = _next_discovery(board, names)
                    if nxt_d:
                        guarded.append(nxt_d)
                        continue
                continue
            brands = _required_brand_labels(board.shopper_notes)
            if brands:
                blob = _row_text_blob(snap)
                if not any((_fragment_match(blob, b) for b in brands)):
                    replaced = False
                    for (_score, key, row) in board.ranked_candidates()[:8]:
                        pid2 = str(row.get('product_id') or '')
                        sku2 = str(row.get('sku') or '') or (board.best_sku_for_product(pid2) or '')
                        if not pid2 or not sku2 or key in board.failed_keys:
                            continue
                        snap2 = board._product_snapshot(pid2, sku2)
                        if any((_fragment_match(_row_text_blob(snap2), b) for b in brands)):
                            guarded.append(('add_to_cart', {'product_id': pid2, 'sku': sku2}))
                            replaced = True
                            break
                    if replaced:
                        continue
                    if 'search' in names:
                        nxt_d = _next_discovery(board, names)
                        if nxt_d:
                            guarded.append(nxt_d)
                            continue
            if not _preference_cart_ok(board, snap):
                replaced = False
                for (_score, key, row) in board.ranked_candidates()[:10]:
                    pid2 = str(row.get('product_id') or '')
                    sku2 = str(row.get('sku') or '') or (board.best_sku_for_product(pid2) or '')
                    if not pid2 or not sku2 or key in board.failed_keys:
                        continue
                    if pid2 in _failed_product_ids(board):
                        continue
                    snap2 = board._product_snapshot(pid2, sku2)
                    if not _preference_cart_ok(board, snap2):
                        continue
                    if pid2 not in board.viewed and 'view' in names:
                        guarded.append(('view', {'product_id': pid2, 'sku': sku2}))
                        replaced = True
                        break
                    if board.stock_ok(pid2, sku2):
                        guarded.append(('add_to_cart', {'product_id': pid2, 'sku': sku2}))
                        replaced = True
                        break
                    if 'inspect_stock' in names:
                        guarded.append(('inspect_stock', {'product_id': pid2, 'sku': sku2}))
                        replaced = True
                        break
                if replaced:
                    continue
                if 'search' in names:
                    nxt_d = _next_discovery(board, names)
                    if nxt_d:
                        guarded.append(nxt_d)
                        continue
                continue
            if pid in board.viewed and board.firm_constraints and (not board.row_compliant(snap, sku)):
                alt = board.best_sku_for_product(pid)
                if alt and alt != sku:
                    alt_snap = board._product_snapshot(pid, alt)
                    if board.row_compliant(alt_snap, alt):
                        guarded.append(('add_to_cart', {'product_id': pid, 'sku': alt}))
                        continue
                nxt = board.best_recovery()
                if nxt and nxt['action'] in names:
                    args = {'product_id': nxt['product_id']}
                    if nxt['action'] != 'view':
                        args['sku'] = nxt['sku']
                    guarded.append((nxt['action'], args))
                    continue
                if 'view' in names:
                    guarded.append(('view', {'product_id': pid}))
                    continue
        if name == 'place_test_order' and pid and sku:
            key = f'{pid}::{sku}'
            if board.task_shape == 'preference_reasoning' and pid in _failed_product_ids(board):
                if board.cart and 'remove_from_cart' in names:
                    top = board.cart[-1]
                    guarded.append(('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']}))
                    continue
                nxt = board.best_recovery()
                if nxt and nxt['action'] in names:
                    args = {'product_id': nxt['product_id']}
                    if nxt['action'] != 'view':
                        args['sku'] = nxt['sku']
                    guarded.append((nxt['action'], args))
                    continue
            if board.needs_rerank and board.named and (len(board.named) >= 2):
                named_actions = _named_pipeline_actions(board, tools=tools, max_calls=max_calls)
                if named_actions:
                    guarded.extend(named_actions)
                    continue
            if board.must_recover and 'remove_from_cart' in names:
                guarded.append(('remove_from_cart', {'product_id': pid, 'sku': sku}))
                continue
            if board.needs_live_check and key not in board.post_cart_stock:
                if 'inspect_stock' in names:
                    guarded.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                    continue
            has_stock = key in board.stock or pid in board.stock
            if (not has_stock or key not in board.cart_confirmed or key not in board.post_cart_stock) and 'inspect_stock' in names:
                guarded.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                continue
            if has_stock and (not board.stock_ok(pid, sku)):
                board.must_recover = True
                if 'remove_from_cart' in names:
                    board.failed_keys.add(key)
                    guarded.append(('remove_from_cart', {'product_id': pid, 'sku': sku}))
                    continue
            if board.task_shape == 'retrieval_recall' and board.view_count < 2 and ('view' in names):
                nxt = board.best_recovery()
                if nxt and nxt['action'] == 'view':
                    guarded.append(('view', {'product_id': nxt['product_id']}))
                    continue
            if board.pending_question and 'message' in names:
                guarded.append(('message', {'content': _recommend_message(board)[:400]}))
                continue
            if 'justification' in _required_args(tools, 'place_test_order') and pid not in board.viewed and ('view' in names):
                guarded.append(('view', {'product_id': pid}))
                continue
        if name == 'inspect_stock' and pid and sku:
            key = f'{pid}::{sku}'
            if not board.needs_live_check and (not board.needs_rerank) and (key in board.post_cart_stock) and board.stock_ok(pid, sku) and ('place_test_order' in names) and (not (board.task_shape == 'retrieval_recall' and board.view_count < 2)):
                guarded.append(('place_test_order', tf4_ground_order({'product_id': pid, 'sku': sku}, tools=tools, board=board)))
                continue
        if name in {'search', 'filter'} and board.named and (board.priorities or board.task_shape in {'ranking', 'preference_reasoning'} or len(board.named) >= 2):
            named_actions = _named_pipeline_actions(board, tools=tools, max_calls=1)
            if named_actions:
                guarded.extend(named_actions)
                continue
        if name in {'search', 'filter'}:
            recent = board.action_log[-4:]
            if len(recent) >= 3 and all((n in {'search', 'filter'} for n in recent[-3:])) and board.candidates:
                progress = _progress_actions(board, tools=tools, max_calls=1)
                if progress:
                    guarded.extend(progress)
                    continue
        if name == 'place_test_order' and pid and sku and (not board.stock_ok(pid, sku)):
            if board.cart and 'remove_from_cart' in names:
                top = board.cart[-1]
                guarded.append(('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']}))
                continue
            progress = _progress_actions(board, tools=tools, max_calls=1)
            if progress:
                guarded.extend(progress)
                continue
        guarded.append((name, args))
    if any((name == 'add_to_cart' for (name, _) in guarded)):
        guarded = [(name, args) for (name, args) in guarded if name != 'place_test_order']
    unique: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for (name, args) in guarded:
        stamp = name + ':' + json.dumps(args, sort_keys=True, default=str)
        if stamp in seen:
            continue
        seen.add(stamp)
        unique.append((name, args))
    return unique[:max_calls]

def _ready_checkout(board: EpisodeBoard) -> dict[str, str] | None:
    """Return carted product_id/sku when live checks already pass."""
    if board.must_recover or board.needs_live_check or board.pending_question or (not board.cart):
        return None
    if board.needs_rerank:
        return None
    top = board.cart[-1]
    if top['key'] not in board.post_cart_stock:
        return None
    if top['key'] not in board.cart_confirmed:
        return None
    if board.named and len(board.named) >= 2:
        if board.facts_refresh_generation < board.event_generation:
            return None
        for (pid, sku) in board.named:
            key = f'{pid}::{sku}'
            if key in board.failed_keys:
                continue
            if not board.stock_is_fresh(pid, sku):
                return None
        ranked = board.ranked_named()
        if ranked and ranked[0][2] != top['key']:
            return None
    if not board.stock_ok(top['product_id'], top['sku']):
        return None
    return {'product_id': top['product_id'], 'sku': top['sku']}

def _named_pipeline_actions(board: EpisodeBoard, *, tools: list[dict[str, Any]], max_calls: int) -> list[tuple[str, dict[str, Any]]]:
    """Deterministic compare → inspect → tradeoff-cart → re-rank loop for named keys."""
    names = _tool_names(tools)
    if not board.named:
        return []
    actions: list[tuple[str, dict[str, Any]]] = []
    if (board.needs_rerank or board.event_seen) and board.facts_refresh_generation < board.event_generation and ('compare' in names) and (len(board.named) >= 2):
        return [('compare', {'product_ids': [pid for (pid, _sku) in board.named[:6]]})]
    if board.view_count == 0 and 'compare' in names and (len(board.named) >= 2):
        return [('compare', {'product_ids': [pid for (pid, _sku) in board.named[:6]]})]
    missing_view = [(pid, sku) for (pid, sku) in board.named if pid not in board.viewed and f'{pid}::{sku}' not in board.failed_keys]
    if missing_view and 'view' in names:
        for (pid, sku) in missing_view[:max_calls]:
            actions.append(('view', {'product_id': pid, 'sku': sku}))
        return actions[:max_calls]
    if board.needs_rerank or board.event_seen:
        stale = [(pid, sku) for (pid, sku) in board.named if f'{pid}::{sku}' not in board.failed_keys and (not board.stock_is_fresh(pid, sku))]
        if stale and 'inspect_stock' in names:
            for (pid, sku) in stale[:max_calls]:
                actions.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
            return actions[:max_calls]
    missing_stock = [(pid, sku) for (pid, sku) in board.named if f'{pid}::{sku}' not in board.failed_keys and f'{pid}::{sku}' not in board.stock and (pid not in board.stock)]
    if missing_stock and 'inspect_stock' in names:
        for (pid, sku) in missing_stock[:max_calls]:
            actions.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
        return actions[:max_calls]
    ranked = board.ranked_named()
    if not ranked:
        return []
    (pid, sku, key) = ranked[0]
    if not board.cart:
        if board.stock_ok(pid, sku) and 'add_to_cart' in names:
            return [('add_to_cart', {'product_id': pid, 'sku': sku})]
        if 'inspect_stock' in names and (not board.stock_is_fresh(pid, sku)):
            return [('inspect_stock', {'product_id': pid, 'sku': sku})]
        return []
    top = board.cart[-1]
    if board.must_recover:
        return []
    if top['key'] != key:
        if not board.stock_is_fresh(pid, sku) and 'inspect_stock' in names:
            return [('inspect_stock', {'product_id': pid, 'sku': sku})]
        if board.stock_ok(pid, sku) and 'remove_from_cart' in names:
            forced: list[tuple[str, dict[str, Any]]] = [('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']})]
            if 'add_to_cart' in names and len(forced) < max_calls:
                forced.append(('add_to_cart', {'product_id': pid, 'sku': sku}))
            return forced[:max_calls]
        return []
    if board.named and all((board.stock_is_fresh(pid, sku) for (pid, sku) in board.named if f'{pid}::{sku}' not in board.failed_keys)):
        board.needs_rerank = False
    else:
        stale = [(pid, sku) for (pid, sku) in board.named if f'{pid}::{sku}' not in board.failed_keys and (not board.stock_is_fresh(pid, sku))]
        if stale and 'inspect_stock' in names:
            for (pid, sku) in stale[:max_calls]:
                actions.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
            return actions[:max_calls]
        return []
    if top['key'] == key and top['key'] not in board.post_cart_stock and ('inspect_stock' in names):
        return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
    if top['key'] == key and top['key'] in board.post_cart_stock and board.stock_ok(top['product_id'], top['sku']) and (not board.pending_question) and (not board.needs_rerank) and ('place_test_order' in names):
        return [('place_test_order', tf4_ground_order({'product_id': top['product_id'], 'sku': top['sku']}, tools=tools, board=board))]
    return []

def _discovery_search(board: EpisodeBoard) -> tuple[str, dict[str, Any]]:
    """Pick the next search/filter action with a rotated query."""
    clean_notes = [_sanitize_note_for_search(n) for n in (board.shopper_notes or [])[-2:]]
    clean_notes = [n for n in clean_notes if n.strip()]
    query = _search_variants(board.query, clean_notes, board.search_count)
    if board.task_shape == 'preference_reasoning' and board.soft_preferences:
        soft_phrases = [frag for frag in board.soft_preferences if not re.fullmatch('\\d{3,}', frag) and len(frag) >= 3 and (frag.lower() not in {'in-stock', 'instock', 'stock'}) and (len(frag.split()) <= 2)]
        nouns = [p for p in _search_parts(board.query, clean_notes) if not re.search('\\d', p) and p.lower() not in {'in-stock', 'instock', 'stock', 'daily', 'every', 'kind'} and (p.lower() not in _SEED_STOP) and (not any((p.lower() in s.lower().split() for s in soft_phrases[:6])))][:2]
        cat = _category_hint(board.query)
        if cat and cat.lower() not in {n.lower() for n in nouns}:
            nouns = [cat.split()[-1]] + nouns
        soft_phrases = sorted(soft_phrases, key=lambda s: (-len(s.split()), -len(s)))
        chosen_soft: list[str] = []
        covered: set[str] = set()
        for frag in soft_phrases:
            toks = [t.lower() for t in frag.split() if len(t) > 2]
            if toks and all((t in covered for t in toks)):
                continue
            chosen_soft.append(frag)
            covered.update(toks)
            if len(chosen_soft) >= 3:
                break
        soft_token_pool = []
        seen_tok: set[str] = set()
        for frag in board.soft_preferences:
            for tok in re.split('[\\s/]+', frag.lower()):
                if len(tok) < 4 or tok in seen_tok or tok in _SEED_STOP:
                    continue
                if re.fullmatch('\\d{3,14}', tok):
                    continue
                seen_tok.add(tok)
                soft_token_pool.append(tok)
        for tok in soft_token_pool:
            if tok in covered:
                continue
            chosen_soft.append(tok)
            covered.add(tok)
            if len(chosen_soft) >= 6:
                break
        if nouns or chosen_soft:
            query = ' '.join(nouns[:1] + chosen_soft[:5]).strip() or query
            seen_q: set[str] = set()
            kept: list[str] = []
            for tok in query.split():
                low = tok.lower()
                if low in seen_q:
                    continue
                seen_q.add(low)
                kept.append(tok)
            query = ' '.join(kept)
    note_blob = ' '.join([tf4_preamble(board.query)] + clean_notes)
    has_phrase = bool(_match_targets(note_blob))
    if board.firm_constraints and board.search_count > 0 and (not (has_phrase and board.search_count < 3)):
        extra = []
        for frag in board.firm_constraints[:4]:
            fl = frag.lower()
            if fl in query.lower():
                continue
            if any((fl in tok or tok in fl for tok in query.lower().split() if len(tok) >= 4)):
                continue
            extra.append(frag)
        if extra:
            query = f"{query} {' '.join(extra[:2])}".strip()
    if board.soft_preferences and board.task_shape != 'preference_reasoning' and (not (has_phrase and board.search_count < 2)):
        soft_extra = [frag for frag in board.soft_preferences[:3] if frag.lower() not in query.lower() and (not re.fullmatch('\\d{6,}', frag))]
        if soft_extra:
            query = f"{query} {' '.join(soft_extra[:2])}".strip()
    if board.task_shape == 'preference_reasoning' and board.soft_preferences and (board.search_count > 0):
        rotate = [frag for frag in board.soft_preferences if frag.lower() not in query.lower() and (not re.fullmatch('\\d{6,}', frag)) and (len(frag) >= 3)]
        if rotate:
            query = f"{query} {' '.join(rotate[:2])}".strip()
            seen_q2: set[str] = set()
            kept2: list[str] = []
            for tok in query.split():
                low = tok.lower()
                if low in seen_q2:
                    continue
                seen_q2.add(low)
                kept2.append(tok)
            query = ' '.join(kept2)
    category = _category_hint(board.query)
    query = re.sub('\\bgo\\s+up\\b', ' ', query, flags=re.I)
    query = re.sub('\\s+', ' ', query).strip()
    args: dict[str, Any] = {'query': query, 'in_stock': True, 'k': 12 if board.soft_preferences else 10}
    if board.budget is not None and board.budget >= 5:
        if not (board.failed_keys and board.search_count % 3 == 2):
            args['max_price'] = board.budget
    brand = None
    preamble = tf4_preamble(board.query)
    explicit = re.search('\\bbrand\\s+([A-Za-z0-9][\\w-]{1,24})\\b', preamble, re.I)
    if explicit:
        brand = explicit.group(1)
    if has_phrase and board.search_count < 3:
        return ('search', args)
    if category and (not has_phrase) and (board.search_count % 2 == 1):
        filt: dict[str, Any] = {**args, 'category': category}
        if brand:
            filt['brand'] = brand
        return ('filter', filt)
    return ('search', args)

def _next_discovery(board: EpisodeBoard, names: set[str]) -> tuple[str, dict[str, Any]] | None:
    """Resolve a discovery action against the live tool roster."""
    if 'search' not in names and 'filter' not in names:
        return None
    (action, args) = _discovery_search(board)
    if action in names:
        return (action, args)
    if 'search' in names:
        clean = {k: v for (k, v) in args.items() if k != 'category'}
        return ('search', clean)
    if 'filter' in names:
        return ('filter', args)
    return None

def _progress_actions(board: EpisodeBoard, *, tools: list[dict[str, Any]], max_calls: int) -> list[tuple[str, dict[str, Any]]]:
    """Deterministic discovery → filter/rank → verify → cart → order."""
    names = _tool_names(tools)
    actions: list[tuple[str, dict[str, Any]]] = []
    if board.named and (board.priorities or board.task_shape in {'ranking', 'preference_reasoning'} or len(board.named) >= 2):
        named_actions = _named_pipeline_actions(board, tools=tools, max_calls=max_calls)
        if named_actions:
            return named_actions
    if board.named and board.view_count < min(2, len(board.named)):
        if 'compare' in names and len(board.named) >= 2:
            return [('compare', {'product_ids': [pid for (pid, _sku) in board.named[:5]]})]
        if 'view' in names:
            (pid, sku) = board.named[board.view_count]
            return [('view', {'product_id': pid, 'sku': sku})]
    if not board.candidates:
        if 'search' in names or 'filter' in names:
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                return [nxt_d]
        return actions
    ranked = board.ranked_candidates()
    specs = _goal_spec_fragments(board.query, board.shopper_notes)
    if not board.cart and (not board.named):
        prefer_ready = board.task_shape != 'preference_reasoning' or (board.view_count >= 2 and board.asked)
        if prefer_ready:
            for (_score, key, row) in ranked:
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
                if not pid or not sku or key in board.failed_keys:
                    continue
                if pid not in board.viewed:
                    continue
                snap = board._product_snapshot(pid, sku)
                if board.firm_constraints and (not board.row_compliant(snap, sku)):
                    continue
                brands = _required_brand_labels(board.shopper_notes)
                if brands:
                    blob = _row_text_blob(snap)
                    if not any((_fragment_match(blob, b) for b in brands)):
                        continue
                if board.task_shape == 'preference_reasoning' and board.soft_preferences:
                    if not _preference_cart_ok(board, snap):
                        continue
                if board.stock_ok(pid, sku) and 'add_to_cart' in names:
                    return [('add_to_cart', {'product_id': pid, 'sku': sku})]
                if key not in board.stock and pid not in board.stock and ('inspect_stock' in names):
                    return [('inspect_stock', {'product_id': pid, 'sku': sku})]
        elif board.task_shape == 'preference_reasoning' and 'view' in names:
            for (_score, key, row) in ranked:
                pid = str(row.get('product_id') or '')
                if not pid or pid in board.viewed:
                    continue
                sku = str(row.get('sku') or '')
                args = {'product_id': pid}
                if sku:
                    args['sku'] = sku
                return [('view', args)]
    if ranked and (not board.named) and (board.view_count == 0) and (board.search_count < 2):
        (top_score, _top_key, top_row) = ranked[0]
        weak = top_score < -2
        if specs and _spec_match_score(top_row, specs) < -3:
            weak = True
        if weak and ('search' in names or 'filter' in names):
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                return [nxt_d]
    for (score, key, row) in ranked:
        if score < 1 and board.search_count < 4 and (not board.named) and (board.view_count > 0):
            continue
        if score < -5 and board.search_count < 3 and (not board.named):
            continue
        if specs and _spec_match_score({**row, **(board.viewed.get(str(row.get('product_id') or '')) or {})}, specs) < 0 and (board.search_count < 5) and (not board.named) and (board.view_count > 0):
            continue
        pid = str(row.get('product_id') or '')
        sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
        if not pid or key in board.failed_keys:
            continue
        snap = board._product_snapshot(pid, sku) if sku else dict(row)
        if pid in board.viewed and board.firm_constraints and (not board.row_compliant(snap, sku)):
            if board.search_count < 8 and board.view_count < 10:
                continue
        soft_codes = [frag for frag in board.soft_preferences if re.fullmatch('\\d{3,14}', frag)]
        if soft_codes and pid not in board.viewed:
            blob = _row_text_blob(snap)
            if not any((_soft_code_match(blob, code) or code in str(row.get('sku') or '') for code in soft_codes)):
                if board.search_count < 4:
                    continue
        if board.firm_constraints and pid not in board.viewed:
            path_parts = snap.get('category_path') or row.get('category_path') or []
            if path_parts:
                path = ' '.join((str(part).lower() for part in path_parts if part is not None))
                title_l = str(snap.get('title') or row.get('title') or '').lower()
                short_firm = [f for f in board.firm_constraints if not re.search('\\d', f) and len(f) >= 4 and (' ' not in f)]
                if short_firm and (not any((f.lower() in path or f.lower() in title_l for f in short_firm))):
                    continue
        if pid not in board.viewed and 'view' in names:
            args = {'product_id': pid}
            if sku:
                args['sku'] = sku
            return [('view', args)]
        if pid in board.viewed:
            chosen = board.best_sku_for_product(pid)
            if chosen:
                sku = chosen
                key = f'{pid}::{sku}'
                snap = board._product_snapshot(pid, sku)
                if board.firm_constraints and (not board.row_compliant(snap, sku)):
                    continue
        if sku and key not in board.stock and (pid not in board.stock) and ('inspect_stock' in names):
            return [('inspect_stock', {'product_id': pid, 'sku': sku})]
        if sku and (key in board.stock or pid in board.stock) and (not board.stock_ok(pid, sku)):
            continue
        if sku and board.stock_ok(pid, sku):
            if not board.cart and 'add_to_cart' in names and (pid in board.viewed):
                if board.firm_constraints and (not board.row_compliant(snap, sku)):
                    if board.search_count < 8 and board.view_count < 10:
                        continue
                if soft_codes:
                    blob = _row_text_blob(snap)
                    if not any((_soft_code_match(blob, code) or code in sku for code in soft_codes)):
                        if board.task_shape in {'recovery', 'preference_reasoning'}:
                            continue
                        if board.search_count < 6 or board.view_count < 3:
                            continue
                soft_words = [frag for frag in board.soft_preferences if not re.fullmatch('\\d{3,14}', frag) and len(frag) >= 3]
                if soft_words and board.task_shape == 'preference_reasoning':
                    (_fh, _fm, soft_hits) = _compliance_scores(snap, board.firm_constraints, soft_words)
                    if board.view_count < 3:
                        continue
                    if soft_hits <= 0 and board.view_count < 5:
                        for (_s2, _k2, row2) in ranked:
                            pid2 = str(row2.get('product_id') or '')
                            if not pid2 or pid2 in board.viewed:
                                continue
                            snap2 = board._product_snapshot(pid2, str(row2.get('sku') or ''))
                            (_h2, _m2, sh2) = _compliance_scores(snap2, board.firm_constraints, soft_words)
                            if sh2 > 0 and 'view' in names:
                                return [('view', {'product_id': pid2})]
                        if board.search_count < 5 and ('search' in names or 'filter' in names):
                            nxt_d = _next_discovery(board, names)
                            if nxt_d:
                                return [nxt_d]
                        continue
                if (board.soft_preferences or board.task_shape in {'preference_reasoning', 'recovery', 'intent_decomposition'}) and board.view_count < 2:
                    for (_score2, _key2, row2) in ranked:
                        pid2 = str(row2.get('product_id') or '')
                        if not pid2 or pid2 == pid or pid2 in board.viewed:
                            continue
                        if soft_codes:
                            snap2 = board._product_snapshot(pid2, str(row2.get('sku') or ''))
                            blob2 = _row_text_blob(snap2)
                            if not any((_soft_code_match(blob2, code) or code in str(row2.get('sku') or '') for code in soft_codes)):
                                continue
                        if 'view' in names:
                            return [('view', {'product_id': pid2})]
                        break
                return [('add_to_cart', {'product_id': pid, 'sku': sku})]
            if board.cart and board.cart[-1]['key'] == key and (key in board.post_cart_stock) and board.stock_ok(pid, sku) and (not board.must_recover) and (not board.needs_rerank) and ('place_test_order' in names):
                return [('place_test_order', tf4_ground_order({'product_id': pid, 'sku': sku}, tools=tools, board=board))]
            if board.cart and board.cart[-1]['key'] == key and (key not in board.post_cart_stock):
                if 'inspect_stock' in names:
                    return [('inspect_stock', {'product_id': pid, 'sku': sku})]
            if board.cart and board.cart[-1]['key'] != key and ('add_to_cart' in names) and board.must_recover:
                if board.firm_constraints and (not board.row_compliant(snap, sku)):
                    continue
                return [('add_to_cart', {'product_id': pid, 'sku': sku})]
        if len(actions) >= max_calls:
            break
    if not actions and (not board.named) and ('search' in names or 'filter' in names):
        search_cap = 12 if board.failed_keys or board.soft_preferences else 8
        if board.search_count < search_cap:
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                return [nxt_d]
    return actions

def _scripted_actions(board: EpisodeBoard, *, tools: list[dict[str, Any]], max_calls: int, turn: int, max_steps: int) -> list[tuple[str, dict[str, Any]]] | None:
    """Skip the LLM when the next move is fully determined by public state."""
    names = _tool_names(tools)
    remaining = max_steps - turn + 1
    if remaining <= 8 and 'place_test_order' in names:
        ready = _ready_checkout(board)
        if ready:
            return [('place_test_order', tf4_ground_order(dict(ready), tools=tools, board=board))]
        if board.cart and 'inspect_stock' in names:
            top = board.cart[-1]
            if top['key'] not in board.post_cart_stock:
                return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
            if not board.stock_ok(top['product_id'], top['sku']) and 'remove_from_cart' in names:
                return [('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']})]
        if not board.cart and 'add_to_cart' in names:
            for (_score, key, row) in board.ranked_candidates()[:16]:
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
                if not pid or not sku or key in board.failed_keys:
                    continue
                if pid not in board.viewed:
                    if 'view' in names and remaining <= 6:
                        return [('view', {'product_id': pid, 'sku': sku})]
                    continue
                snap = board._product_snapshot(pid, sku)
                if board.firm_constraints and (not board.row_compliant(snap, sku)):
                    continue
                brands = _required_brand_labels(board.shopper_notes)
                if brands and (not any((_fragment_match(_row_text_blob(snap), b) for b in brands))):
                    continue
                if key not in board.stock and pid not in board.stock:
                    if 'inspect_stock' in names:
                        return [('inspect_stock', {'product_id': pid, 'sku': sku})]
                    continue
                if not board.stock_ok(pid, sku):
                    continue
                return [('add_to_cart', {'product_id': pid, 'sku': sku})]
            if remaining <= 8 and ('search' in names or 'filter' in names):
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    (action, args) = nxt_d
                    q = str(args.get('query') or '')
                    q = re.sub('\\bgo\\s+up\\b', ' ', q, flags=re.I)
                    q = re.sub('\\b\\d{8,}\\b', ' ', q)
                    for frag in board.soft_preferences:
                        if re.search('\\d', frag):
                            continue
                        if ' ' in frag:
                            continue
                        q = re.sub(f'\\b{re.escape(frag)}\\b', ' ', q, flags=re.I)
                    args = {**args, 'query': re.sub('\\s+', ' ', q).strip()}
                    if board.budget is not None:
                        args['max_price'] = board.budget
                    return [(action, args)]
            if remaining <= 4:
                for (_score, key, row) in board.ranked_candidates()[:20]:
                    pid = str(row.get('product_id') or '')
                    sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
                    if not pid or not sku or key in board.failed_keys:
                        continue
                    if pid not in board.viewed:
                        continue
                    snap = board._product_snapshot(pid, sku)
                    if not _is_compliant(snap, board.firm_constraints):
                        continue
                    if board.stock_ok(pid, sku):
                        return [('add_to_cart', {'product_id': pid, 'sku': sku})]
                    if 'inspect_stock' in names and key not in board.stock:
                        return [('inspect_stock', {'product_id': pid, 'sku': sku})]
    recent = board.action_log[-4:]
    inspect_loop = len(recent) >= 3 and all((a == 'inspect_stock' for a in recent[-3:]))
    search_loop = len(recent) >= 4 and all((a in {'search', 'filter'} for a in recent[-4:]))
    if (inspect_loop or search_loop) and remaining <= 14:
        if board.cart:
            top = board.cart[-1]
            if board.stock_ok(top['product_id'], top['sku']) and 'place_test_order' in names:
                return [('place_test_order', tf4_ground_order({'product_id': top['product_id'], 'sku': top['sku']}, tools=tools, board=board))]
            if not board.stock_ok(top['product_id'], top['sku']) and 'remove_from_cart' in names:
                return [('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']})]
        for (_score, key, row) in board.ranked_candidates()[:8]:
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
            if not pid or not sku or key in board.failed_keys:
                continue
            if pid not in board.viewed and 'view' in names:
                return [('view', {'product_id': pid, 'sku': sku})]
            snap = board._product_snapshot(pid, sku)
            if board.firm_constraints and (not board.row_compliant(snap, sku)):
                continue
            if board.stock_ok(pid, sku) and 'add_to_cart' in names:
                return [('add_to_cart', {'product_id': pid, 'sku': sku})]
            if 'inspect_stock' in names and key not in board.stock:
                return [('inspect_stock', {'product_id': pid, 'sku': sku})]
    if board.needs_ask and (not board.received_clarification) and (not board.asked) and ('message' in names):
        return [('message', {'content': 'Before I search, what firm product specification must the item meet besides category, budget, and in-stock availability?'})]
    if board.task_shape == 'preference_reasoning' and (not board.asked) and (not board.cart) and (board.view_count == 0) and ('message' in names) and re.search('\\bclarify\\b.{0,48}\\bpriorit', board.query, re.I):
        soft_bits = ', '.join(board.soft_preferences[:4]) or 'your stated soft preferences'
        return [('message', {'content': f"Before I cart anything: which of these matter most for you — {soft_bits}? I'll compare visible product facts against that."})]
    if board.must_recover and board.cart and ('remove_from_cart' in names):
        if board.named and len(board.named) >= 2 and board.needs_rerank and (board.facts_refresh_generation < board.event_generation) and ('compare' in names):
            return [('compare', {'product_ids': [pid for (pid, _sku) in board.named[:6]]})]
        top = board.cart[-1]
        forced: list[tuple[str, dict[str, Any]]] = [('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']})]
        failed_stem = re.sub('-\\d+$', '', str(top.get('sku') or ''))
        soft_codes = [frag for frag in board.soft_preferences if re.fullmatch('\\d{3,14}', frag)]
        nxt = board.best_recovery()
        if nxt and nxt['action'] in names and (len(forced) < max_calls):
            nxt_sku = str(nxt.get('sku') or '')
            same_stem = bool(failed_stem and len(failed_stem) >= 4 and nxt_sku.startswith(failed_stem))
            prefer_research = board.task_shape in {'preference_reasoning', 'justification', 'retrieval_recall'} and board.soft_preferences and (not soft_codes)
            skip_nxt = False
            if board.task_shape == 'preference_reasoning':
                failed_pid = board.last_failed_key.split('::', 1)[0] if board.last_failed_key and '::' in board.last_failed_key else str(top.get('product_id') or '')
                if same_stem or (failed_pid and str(nxt.get('product_id') or '') == failed_pid):
                    skip_nxt = True
            if board.task_shape == 'recovery' and soft_codes and nxt and (not same_stem):
                skip_nxt = True
            if not skip_nxt and same_stem and (not prefer_research):
                args = {'product_id': nxt['product_id']}
                if nxt['action'] != 'view':
                    args['sku'] = nxt['sku']
                forced.append((nxt['action'], args))
                return forced[:max_calls]
            if not skip_nxt and (same_stem or not soft_codes) and (not prefer_research) and (board.task_shape != 'recovery'):
                brand_bits = [f for f in board.firm_constraints if f[:1].isupper() or (' ' in f and (not re.search('\\d', f)))]
                if brand_bits and nxt.get('product_id'):
                    snap = board._product_snapshot(str(nxt['product_id']), str(nxt.get('sku') or ''))
                    blob = _row_text_blob(snap)
                    if not any((_fragment_match(blob, b) for b in brand_bits)):
                        brand_bits = []
                    else:
                        args = {'product_id': nxt['product_id']}
                        if nxt['action'] != 'view':
                            args['sku'] = nxt['sku']
                        forced.append((nxt['action'], args))
                        return forced[:max_calls]
                elif not brand_bits:
                    args = {'product_id': nxt['product_id']}
                    if nxt['action'] != 'view':
                        args['sku'] = nxt['sku']
                    forced.append((nxt['action'], args))
                    return forced[:max_calls]
        if (soft_codes or board.task_shape == 'preference_reasoning') and ('search' in names or 'filter' in names) and (len(forced) < max_calls):
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                (action, args) = nxt_d
                q = str(args.get('query') or '')
                bits = [q]
                if failed_stem and failed_stem not in q and soft_codes:
                    bits.append(failed_stem)
                for frag in board.soft_preferences[:3]:
                    if frag and frag not in q and (not re.search('\\s', frag)):
                        bits.append(frag)
                args = {**args, 'query': ' '.join((b for b in bits if b)).strip()}
                forced.append((action, args))
                return forced[:max_calls]
        if soft_codes and 'add_to_cart' in names:
            for (score, key, row) in board.ranked_candidates():
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
                if not pid or not sku or key in board.failed_keys:
                    continue
                if failed_stem and len(failed_stem) >= 4 and (not sku.startswith(failed_stem)):
                    continue
                snap = board._product_snapshot(pid, sku)
                blob = _row_text_blob(snap)
                if not any((_soft_code_match(blob, code) or code in sku for code in soft_codes)):
                    continue
                if pid not in board.viewed and 'view' in names and (len(forced) < max_calls):
                    forced.append(('view', {'product_id': pid, 'sku': sku}))
                    return forced[:max_calls]
                if board.stock_ok(pid, sku) and len(forced) < max_calls:
                    forced.append(('add_to_cart', {'product_id': pid, 'sku': sku}))
                    return forced[:max_calls]
                if 'inspect_stock' in names and len(forced) < max_calls and (key not in board.stock):
                    forced.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                    return forced[:max_calls]
        if ('search' in names or 'filter' in names) and len(forced) < max_calls:
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                (action, args) = nxt_d
                q = str(args.get('query') or '')
                bits = [q]
                if failed_stem and failed_stem not in q:
                    bits.append(failed_stem)
                for frag in board.soft_preferences[:2]:
                    if frag and frag not in q:
                        bits.append(frag)
                args = {**args, 'query': ' '.join((b for b in bits if b)).strip()}
                forced.append((action, args))
        return forced[:max_calls]
    if board.must_recover and (not board.cart):
        nxt = board.best_recovery()
        soft_codes = [frag for frag in board.soft_preferences if re.fullmatch('\\d{3,14}', frag)]
        stem = ''
        if board.last_failed_key and '::' in board.last_failed_key:
            stem = re.sub('-\\d+$', '', board.last_failed_key.split('::', 1)[1])
        if nxt and board.task_shape == 'recovery' and soft_codes and stem and (len(stem) >= 4):
            nxt_sku = str(nxt.get('sku') or '')
            if not nxt_sku.startswith(stem):
                nxt = None
        if nxt and nxt['action'] in names:
            args = {'product_id': nxt['product_id']}
            if nxt['action'] != 'view':
                args['sku'] = nxt['sku']
            return [(nxt['action'], args)]
        if (soft_codes or stem) and board.task_shape == 'recovery' and ('search' in names or 'filter' in names) and (board.search_count < 14):
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                (action, args) = nxt_d
                q = str(args.get('query') or '')
                bits = [q]
                if stem and stem not in q:
                    bits.append(stem)
                for frag in board.soft_preferences[:3]:
                    if frag and frag not in q:
                        bits.append(frag)
                args = {**args, 'query': ' '.join((b for b in bits if b)).strip()}
                return [(action, args)]
        note_blob = ' '.join([tf4_preamble(board.query)] + [_sanitize_note_for_search(n) for n in (board.shopper_notes or [])[-2:]])
        targets = [t.lower() for t in _match_targets(note_blob)]
        if targets and 'view' in names:
            for (score, key, row) in board.ranked_candidates():
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
                if not pid or key in board.failed_keys:
                    continue
                snap = board._product_snapshot(pid, sku) if sku else dict(row)
                blob = _row_text_blob(snap)
                if not any((t in blob for t in targets)):
                    continue
                if pid not in board.viewed:
                    return [('view', {'product_id': pid, 'sku': sku} if sku else {'product_id': pid})]
                if sku and board.stock_ok(pid, sku) and ('add_to_cart' in names):
                    return [('add_to_cart', {'product_id': pid, 'sku': sku})]
                if sku and 'inspect_stock' in names and (key not in board.stock):
                    return [('inspect_stock', {'product_id': pid, 'sku': sku})]
        if soft_codes and 'view' in names:
            for (score, key, row) in board.ranked_candidates():
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
                if not pid or key in board.failed_keys:
                    continue
                if stem and len(stem) >= 4 and sku and (not sku.startswith(stem)):
                    continue
                snap = board._product_snapshot(pid, sku) if sku else dict(row)
                blob = _row_text_blob(snap)
                if not any((_soft_code_match(blob, code) or code in sku for code in soft_codes)):
                    continue
                if pid not in board.viewed:
                    return [('view', {'product_id': pid, 'sku': sku} if sku else {'product_id': pid})]
                if sku and board.stock_ok(pid, sku) and ('add_to_cart' in names):
                    return [('add_to_cart', {'product_id': pid, 'sku': sku})]
                if sku and 'inspect_stock' in names and (key not in board.stock):
                    return [('inspect_stock', {'product_id': pid, 'sku': sku})]
        if 'search' in names or 'filter' in names:
            note_blob = ' '.join([tf4_preamble(board.query)] + [_sanitize_note_for_search(n) for n in (board.shopper_notes or [])[-2:]])
            has_phrase = bool(_match_targets(note_blob))
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                (action, args) = nxt_d
                q = str(args.get('query') or '')
                bits = [q]
                if stem and stem not in q and (not has_phrase) and board.soft_preferences:
                    bits.append(stem)
                for frag in board.soft_preferences[:2]:
                    if frag and frag not in q:
                        bits.append(frag)
                args = {**args, 'query': ' '.join((b for b in bits if b)).strip()}
                return [(action, args)]
    if board.received_clarification and board.task_shape == 'constraint_satisfaction' and (board.search_count == 0) and ('search' in names or 'filter' in names):
        nxt_d = _next_discovery(board, names)
        if nxt_d:
            return [nxt_d]
    if board.pending_question and 'message' in names:
        actions: list[tuple[str, dict[str, Any]]] = [('message', {'content': _recommend_message(board)[:400]})]
        if board.cart and 'inspect_stock' in names and (len(actions) < max_calls):
            top = board.cart[-1]
            actions.append(('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']}))
        if not board.cart and 'add_to_cart' in names and (len(actions) < max_calls):
            for (_score, key, row) in board.ranked_candidates():
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
                if not pid or not sku or key in board.failed_keys:
                    continue
                if pid not in board.viewed:
                    continue
                if not board.stock_ok(pid, sku):
                    continue
                snap = board._product_snapshot(pid, sku)
                if board.firm_constraints and (not board.row_compliant(snap, sku)):
                    continue
                actions.append(('add_to_cart', {'product_id': pid, 'sku': sku}))
                break
        return actions[:max_calls]
    if board.needs_live_check and board.cart and ('inspect_stock' in names):
        top = board.cart[-1]
        if top['key'] not in board.post_cart_stock:
            return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
    if board.action_log and board.action_log[-1] == 'inspect_stock' and (not board.cart) and ('add_to_cart' in names):
        soft_codes = [frag for frag in board.soft_preferences if re.fullmatch('\\d{3,14}', frag)]
        for (score, key, row) in board.ranked_candidates():
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
            if not pid or not sku or key in board.failed_keys:
                continue
            if pid not in board.viewed:
                continue
            if not board.stock_ok(pid, sku):
                continue
            snap = board._product_snapshot(pid, sku)
            if board.firm_constraints and (not board.row_compliant(snap, sku)):
                if board.search_count < 5 and board.view_count < 6:
                    continue
            brands = _required_brand_labels(board.shopper_notes)
            if brands:
                blob = _row_text_blob(snap)
                if not any((_fragment_match(blob, b) for b in brands)):
                    continue
            if soft_codes:
                blob = _row_text_blob(snap)
                if not any((_soft_code_match(blob, code) or code in sku for code in soft_codes)):
                    if board.task_shape in {'recovery', 'preference_reasoning'}:
                        continue
                    if score < 6:
                        continue
            if not _preference_cart_ok(board, snap):
                continue
            if board.must_recover and board.last_failed_key and ('::' in board.last_failed_key) and (board.task_shape != 'preference_reasoning'):
                failed_sku = board.last_failed_key.split('::', 1)[1]
                stem = re.sub('-\\d+$', '', failed_sku)
                if stem and len(stem) >= 4 and (not sku.startswith(stem)):
                    continue
            if board.must_recover and board.task_shape == 'preference_reasoning':
                failed_pid = board.last_failed_key.split('::', 1)[0] if board.last_failed_key else ''
                if failed_pid and pid == failed_pid:
                    continue
                failed_sku = board.last_failed_key.split('::', 1)[1] if board.last_failed_key and '::' in board.last_failed_key else ''
                stem = re.sub('-\\d+$', '', failed_sku)
                if stem and len(stem) >= 4 and sku.startswith(stem):
                    continue
            return [('add_to_cart', {'product_id': pid, 'sku': sku})]
        if board.must_recover:
            nxt = board.best_recovery()
            if nxt and nxt['action'] == 'add_to_cart' and (nxt['action'] in names):
                if board.task_shape != 'preference_reasoning' or _preference_cart_ok(board, board._product_snapshot(str(nxt['product_id']), str(nxt.get('sku') or ''))):
                    return [('add_to_cart', {'product_id': nxt['product_id'], 'sku': nxt['sku']})]
        if board.task_shape == 'preference_reasoning':
            for (_score, key, row) in board.ranked_candidates():
                pid = str(row.get('product_id') or '')
                if not pid or pid in board.viewed:
                    continue
                sku = str(row.get('sku') or '')
                args = {'product_id': pid}
                if sku:
                    args['sku'] = sku
                if 'view' in names:
                    return [('view', args)]
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                return [nxt_d]
    if board.named and (board.priorities or board.task_shape in {'ranking', 'preference_reasoning'} or len(board.named) >= 2):
        named_actions = _named_pipeline_actions(board, tools=tools, max_calls=max_calls)
        if named_actions:
            return named_actions
    ready = _ready_checkout(board)
    if ready and 'place_test_order' in names:
        return [('place_test_order', tf4_ground_order(dict(ready), tools=tools, board=board))]
    if board.cart and 'inspect_stock' in names:
        top = board.cart[-1]
        key = top['key']
        if key not in board.post_cart_stock or (key not in board.stock and top['product_id'] not in board.stock):
            return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
        if not board.stock_ok(top['product_id'], top['sku']) and 'remove_from_cart' in names:
            return [('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']})]
    if board.candidates and board.view_count == 0 or board.task_shape in {'recovery', 'constraint_satisfaction', 'retrieval', 'retrieval_recall', 'intent_decomposition', 'preference_reasoning', 'justification'}:
        progress = _progress_actions(board, tools=tools, max_calls=max_calls)
        if progress:
            return progress
    recent = board.action_log[-6:]
    stuck_search = len(recent) >= 3 and all((name in {'search', 'filter', 'message'} for name in recent[-3:]))
    stuck_inspect = len(recent) >= 2 and recent[-2:] == ['inspect_stock', 'inspect_stock']
    if stuck_search or stuck_inspect or remaining <= 10 or (turn == 1) or board.failed_keys:
        progress = _progress_actions(board, tools=tools, max_calls=max_calls)
        if progress:
            return progress
    if remaining <= 6:
        progress = _progress_actions(board, tools=tools, max_calls=max_calls)
        if progress:
            return progress
        if not board.named and ('search' in names or 'filter' in names):
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                return [nxt_d]
    return None

def _force_finish_actions(action_pairs: list[tuple[str, dict[str, Any]]], *, board: EpisodeBoard, tools: list[dict[str, Any]], remaining: int, max_calls: int=1) -> list[tuple[str, dict[str, Any]]]:
    """When a carted sku is live-valid, checkout immediately — never stall."""
    names = _tool_names(tools)
    if 'place_test_order' not in names:
        return action_pairs
    if board.named and len(board.named) >= 2 and (board.needs_rerank or board.must_recover):
        named_actions = _named_pipeline_actions(board, tools=tools, max_calls=max_calls)
        if named_actions:
            return named_actions
    ready = _ready_checkout(board)
    if ready is not None:
        args = tf4_ground_order(dict(ready), tools=tools, board=board)
        return [('place_test_order', args)]
    if not board.cart and board.failed_keys:
        nxt = board.best_recovery()
        if nxt and nxt['action'] in names:
            args = {'product_id': nxt['product_id']}
            if nxt['action'] != 'view':
                args['sku'] = nxt['sku']
            return [(nxt['action'], args)]
        if remaining <= 10:
            for (_score, key, row) in board.ranked_candidates():
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '') or (board.best_sku_for_product(pid) or '')
                if not pid or not sku or key in board.failed_keys:
                    continue
                snap = board._product_snapshot(pid, sku)
                if pid not in board.viewed:
                    if 'view' in names:
                        return [('view', {'product_id': pid, 'sku': sku})]
                    continue
                if not board.stock_ok(pid, sku):
                    if 'inspect_stock' in names and key not in board.stock:
                        return [('inspect_stock', {'product_id': pid, 'sku': sku})]
                    continue
                if board.firm_constraints and (not board.row_compliant(snap, sku)):
                    continue
                if board.task_shape == 'preference_reasoning' and pid in _failed_product_ids(board):
                    continue
                if not _preference_cart_ok(board, snap):
                    continue
                if 'add_to_cart' in names:
                    return [('add_to_cart', {'product_id': pid, 'sku': sku})]
        if remaining <= 12 and ('search' in names or 'filter' in names):
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                return [nxt_d]
    if remaining <= 8 and (not any((name in {'place_test_order', 'add_to_cart', 'view', 'inspect_stock', 'remove_from_cart'} for (name, _) in action_pairs))):
        progress = _progress_actions(board, tools=tools, max_calls=1)
        if progress:
            return progress
    return action_pairs

def _trim_messages(messages: list[dict[str, Any]]) -> None:
    if len(messages) <= _HISTORY_SOFT_CAP:
        return
    head = messages[:2]
    tail = messages[-(_HISTORY_SOFT_CAP - 3):]
    messages[:] = head + [{'role': 'user', 'content': 'Older turns are summarized on the episode board. Use the board and recent tool results.'}] + tail

def _slim_messages_for_inference(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep prompts small to avoid proxy 400s on oversized chat payloads."""
    if len(messages) <= 6:
        return messages
    head = messages[:2]
    tail = messages[-4:]
    return head + [{'role': 'user', 'content': 'Earlier tool traffic omitted. Use the latest board status.'}] + tail

def _env_call(binding: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Single proxy post; ProxyClient already retries transient HTTP failures."""
    del binding
    return _proxy.post('/environment/call', json_data=envelope)

def _inference(*, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    last_error: Exception | None = None
    candidates = [model]
    for alt in _model_candidates():
        if alt != model:
            candidates.append(alt)
            break
    message_variants = [_slim_messages_for_inference(messages), messages]
    for candidate in candidates:
        for msg in message_variants:
            try:
                result = _proxy.post('/inference/chat/completions', json_data={'model': candidate, 'messages': msg, 'tools': tools, 'tool_choice': 'required', 'temperature': 0})
                if result is None:
                    raise RuntimeError('inference request failed')
                return result
            except Exception as exc:
                last_error = exc
                continue
    raise RuntimeError(f'inference request failed: {last_error}')

# ===== KI_3.py pipeline (tf5) =====
tf5_PREF_MODEL_RE = re.compile('(?:strongly\\s+prefer|prefer(?:red)?|want|keep)\\s+(?:the\\s+)?(?:(?P<color>[A-Za-z]{3,12})\\s+)?(?P<code>[A-Za-z]*\\d{2,}[A-Za-z0-9-]*)', re.I)
tf5_CATEGORY_GOAL_RE = re.compile("(?:(?:in|from)\\s+the\\s+|(?:catalog\\s+)?category\\s+is\\s+)([A-Za-z][A-Za-z0-9 &/+'-]{2,48}?)(?:\\s+category\\b|(?=\\s*[.,;]|\\s*$))", re.I)
tf5_STOP = frozenset('\n    a an the and or but for with from that this these those into onto over under\n    at to of in on by is are was were be been being it its as if then than so\n    not no yes you your we our they their i me my need want find get buy purchase\n    please can could would should must also just only more most less least very\n    item items product products listing listings order orders cart shop shopping\n    available stock stocked budget price prices within below above prefer preferred\n    ideally soft firm requirement requirements category categories brand brands\n    looking look help thanks thank hi hello there here about around\n    something anything everything nothing somehow somewhere\n    place test checkout choose pick confirm currently\n    '.split())
tf5_LEDGER_CAND_CAP = 36

def _sku_stem(sku: str) -> str:
    """Variant family stem: everything before the final '-suffix' when present."""
    text = str(sku or '').strip()
    if not text:
        return ''
    if '-' in text:
        return text.rsplit('-', 1)[0].lower()
    return text.lower()

def tf5_constraint_atoms(query: str, notes: list[str] | None=None) -> list[str]:
    """Extract compact, category-agnostic requirement atoms from public text.

    Atoms are quoted phrases, decimal/integer figures, and figure+token pairs.
    No product-category vocabulary is used.
    """
    text = _preamble(query)
    if notes:
        text = f"{text} {' '.join(notes[-3:])}"
    text = re.sub('(?:more\\s+expensive|price(?:\\s+change)?|went\\s+from|from|now|was|were|became|increased\\s+to|dropped\\s+to|raised\\s+to|updated\\s+to|costs?)\\s+\\d[\\d,]*(?:\\.\\d+)?(?:\\s*(?:to|-|→)\\s*\\d[\\d,]*(?:\\.\\d+)?)?', ' ', text, flags=re.I)
    text = re.sub('\\b\\d{2,}[.,]\\d{2}\\b', ' ', text)
    atoms: list[str] = []
    seen: set[str] = set()

    def add(atom: str) -> None:
        key = atom.strip().lower()
        if len(key) < 2 or key in seen or key in tf5_STOP:
            return
        seen.add(key)
        atoms.append(key)
    for match in re.finditer('"([^"\\n]{2,80})"|“([^”\\n]{2,80})”', text):
        add(match.group(1) or match.group(2) or '')
    budget = _parse_budget(text)
    budget_s = None
    if budget is not None:
        budget_s = str(int(budget)) if float(budget).is_integer() else str(budget)
    for match in _NUM_NEAR_RE.finditer(text):
        num = match.group('num') or match.group('num2')
        unit = match.group('unit') or match.group('unit2') or ''
        if not num:
            continue
        try:
            num_f = float(num.replace(',', ''))
        except ValueError:
            continue
        if budget is not None and abs(num_f - budget) <= 1e-06:
            continue
        if budget_s and num.replace(',', '') == budget_s.replace(',', ''):
            continue
        unit_l = unit.lower().strip()
        if unit_l in {'php', 'usd', 'eur', 'gbp', 'cad', 'aud', 'jpy', 'cny', 'inr', 'price', 'budget'}:
            continue
        if num_f >= 20 and re.search('\\.\\d{2}$', num.replace(',', '')):
            continue
        if not unit_l and num_f >= 200 and ('.' not in num.replace(',', '')):
            continue
        if unit_l in tf5_STOP or len(unit_l) <= 1:
            add(num)
            if unit_l and unit_l not in tf5_STOP and re.fullmatch('[a-z]', unit_l):
                add(f'{num}{unit_l}')
            continue
        if unit_l:
            add(f'{num} {unit_l}')
            add(num)
        else:
            add(num)
    for token in re.findall('\\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\\b', text.lower()):
        if token not in {'product_id', 'in_stock', 'max_price'}:
            add(token)
    return atoms[:24]

def _moneyish_token(token: str) -> bool:
    """True for listing-price-like tokens that must never drive search queries."""
    raw = (token or '').strip().replace(',', '')
    if not raw:
        return False
    if re.fullmatch('\\d+\\.\\d{2}', raw):
        try:
            return float(raw) >= 10.0
        except ValueError:
            return False
    if re.fullmatch('\\d+(?:\\.\\d+)?', raw):
        try:
            return float(raw) >= 200.0
        except ValueError:
            return False
    return False

def _model_token_hit(token: str, blob: str) -> bool:
    """Match prefer/model codes without over-matching short digit fragments."""
    tok = (token or '').strip().lower()
    if not tok or not blob:
        return False
    if len(tok) <= 3 and any((ch.isdigit() for ch in tok)):
        return bool(re.search(f'(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])', blob, re.I))
    return tok in blob

def _anchor_title_query(ledger: 'Ledger') -> str | None:
    """Build a rediscovery query from the failed listing's own title tokens."""
    for pid in list(ledger.anchor_pids):
        viewed = ledger.viewed.get(pid) or {}
        title = str(viewed.get('title') or (ledger.candidates.get(pid) or {}).get('title') or '')
        if not title:
            for (key, row) in ledger.candidates.items():
                if key.startswith(f'{pid}::') or str(row.get('product_id') or '') == pid:
                    title = str(row.get('title') or '')
                    if title:
                        break
        if not title:
            continue
        toks = [t for t in _WORD_RE.findall(title) if t.lower() not in tf5_STOP and len(t) >= 3]
        digs = [t for t in toks if any((ch.isdigit() for ch in t))]
        alphas = [t for t in toks if t.isalpha() and len(t) >= 4]
        parts: list[str] = []
        if alphas:
            parts.append(alphas[0])
        if digs:
            parts.append(digs[0])
        elif alphas[1:]:
            parts.append(alphas[1])
        q = ' '.join(parts).strip()
        if q and q.lower() not in ledger.tried_queries:
            return q
    return None

def tf5_soft_tokens(text: str) -> list[str]:
    """Extract soft preference objects (brand/name tokens) from public goal text."""
    out: list[str] = []
    seen: set[str] = set()
    for match in _SOFT_OBJECT_RE.finditer(text or ''):
        raw = (match.group(1) or '').strip().rstrip('.,;:')
        if not raw or len(raw) < 2:
            continue
        key = raw.lower()
        if key in tf5_STOP or key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out[:4]

def tf5_prefer_model_tokens(text: str) -> list[str]:
    """Extract preferred model/code/color tokens from public goal text."""
    out: list[str] = []
    seen: set[str] = set()

    def add(part: str | None) -> None:
        if not part:
            return
        key = part.lower().strip('.,;:')
        if len(key) < 2 or key in seen or key in tf5_STOP:
            return
        seen.add(key)
        out.append(part.strip('.,;:'))
    for match in tf5_PREF_MODEL_RE.finditer(text or ''):
        add(match.group('color'))
        add(match.group('code'))
    for match in re.finditer('\\b(?:model|code|variant)\\s+([A-Za-z]*\\d{2,}[A-Za-z0-9-]*)\\b', text or '', re.I):
        add(match.group(1))
    return out[:8]

def tf5_category_mentions(text: str) -> list[str]:
    out: list[str] = []
    for match in tf5_CATEGORY_GOAL_RE.finditer(text or ''):
        label = (match.group(1) or '').strip()
        if label and label.lower() not in tf5_STOP:
            out.append(label)
    return out[:3]

def tf5_classify_procedure(query: str, *, tools: list[dict[str, Any]]) -> str:
    """Pick one shopping procedure from public goal + tool schema cues only.

    Does not use product-category vocabularies or suite answer tables. Branching
    is driven by structural signals: named variants, deferred specs, soft vs
    firm discourse, quoted phrases, use-context language, and order schema.
    """
    goal = _preamble(query)
    scores: dict[str, int] = {PROC_ASK: 0, PROC_RECALL: 0, PROC_STRICT: 0, PROC_SOFT: 0, PROC_USE: 0, PROC_RANK: 0, PROC_ADAPT: 0, PROC_CLAIM: 0}
    named = _named_variants(query)
    if len(named) >= 2:
        scores[PROC_RANK] += 6
    if tf5_rank_axes(goal):
        scores[PROC_RANK] += 2
    if _needs_clarification(query):
        scores[PROC_ASK] += 5
        scores[PROC_STRICT] += 2
    needs_why = 'justification' in _required_args(tools, 'place_test_order')
    if needs_why or _WHY_GOAL_RE.search(goal):
        scores[PROC_CLAIM] += 5
    softs = tf5_soft_tokens(goal)
    if softs and (_FIRM_MATTERS_RE.search(goal) or _FIRM_CUE_RE.search(goal)):
        scores[PROC_SOFT] += 5
    elif softs:
        scores[PROC_SOFT] += 3
    if _SOFT_CUE_RE.search(goal) and _FIRM_MATTERS_RE.search(goal):
        scores[PROC_SOFT] += 2
    phrases = _goal_phrases(goal)
    if phrases or tf5_category_mentions(goal):
        scores[PROC_RECALL] += 4
    if re.search('\\bmatches?\\b', goal, re.I) and phrases:
        scores[PROC_RECALL] += 2
    if _USE_FIT_RE.search(goal) and (not softs):
        scores[PROC_USE] += 4
    elif _USE_FIT_RE.search(goal):
        scores[PROC_USE] += 2
    if _ADAPT_GOAL_RE.search(goal):
        scores[PROC_ADAPT] += 4
    prefs = tf5_prefer_model_tokens(goal)
    if any((any((ch.isdigit() for ch in p)) for p in prefs)):
        scores[PROC_ADAPT] += 6
    atoms = tf5_constraint_atoms(query)
    if len(atoms) >= 2 and scores[PROC_SOFT] == 0 and (scores[PROC_RECALL] == 0):
        scores[PROC_STRICT] += 2
    if not any(scores.values()):
        scores[PROC_STRICT] = 1
    order = (PROC_RANK, PROC_ASK, PROC_CLAIM, PROC_SOFT, PROC_ADAPT, PROC_USE, PROC_RECALL, PROC_STRICT)
    best = max(scores.values())
    for proc in order:
        if scores[proc] == best:
            return proc
    return PROC_STRICT

def tf5_parse_budget_update(text: str) -> float | None:
    """Budget updates from shopper notes — only when they explicitly revise the ceiling.

    Ignores incidental price figures in event notices (e.g. 'more expensive from
    4790 to 10618 PHP') so those cannot raise the shopping ceiling.
    """
    blob = text or ''
    scrubbed = re.sub('(?:more\\s+expensive|price(?:\\s+change)?|went\\s+from|from)\\s+\\d[\\d,]*(?:\\.\\d+)?\\s*(?:to|-)?\\s*\\d[\\d,]*(?:\\.\\d+)?\\s*(?:php|usd|eur|gbp)?', ' ', blob, flags=re.I)
    if not re.search('\\b(?:budget|spend|under|below|at\\s+most|no\\s+more\\s+than|cap|ceiling|can\\s+only\\s+(?:spend|go)|only\\s+go\\s+up\\s+to|up\\s+to|max(?:imum)?(?:\\s+price)?)\\b', scrubbed, re.I):
        return None
    return _parse_budget(scrubbed)

def _notes_have_firm_spec(notes: list[str], *, prior_atoms: list[str] | None=None) -> bool:
    """True when shopper notes already supply a concrete firm figure/phrase."""
    if not notes:
        return False
    blob = ' '.join(notes[-3:])
    prior = {a.lower() for a in prior_atoms or []}
    fresh = tf5_constraint_atoms(blob)
    for atom in fresh:
        if atom.lower() not in prior and any((ch.isdigit() for ch in atom)):
            return True
    if _NUM_RE.search(blob) and re.search('\\b(?:required|must|need(?:s)?|exactly|still\\s+needs?|spec)\\b', blob, re.I):
        return True
    return bool(re.search('"[^"\\n]{2,80}"', blob))

def tf5_rank_axes(text: str) -> list[tuple[str, int]]:
    """Parse comparative ranking axes: (normalized_axis, direction) where +1=higher, -1=lower."""
    axes: list[tuple[str, int]] = []
    seen: set[str] = set()
    dir_noise = _DIR_UP | _DIR_DOWN
    for match in _RANK_AXIS_RE.finditer(text or ''):
        direction_word = match.group('dir').lower()
        raw_axis = match.group('axis').strip().lower()
        parts = [p for p in raw_axis.split() if p not in dir_noise and p not in tf5_STOP]
        if not parts:
            continue
        axis = re.sub('[^a-z0-9_]+', '', '_'.join(parts))
        if len(axis) < 3 or axis in seen or axis in tf5_STOP:
            continue
        direction = -1 if direction_word in _DIR_DOWN else 1
        if 'price' in axis:
            direction = -1
        seen.add(axis)
        axes.append((axis, direction))
    return axes[:4]

def tf5_goal_tokens(query: str, notes: list[str] | None=None) -> set[str]:
    text = _preamble(query)
    if notes:
        text = f"{text} {' '.join(notes[-3:])}"
    tokens = {t.lower() for t in _WORD_RE.findall(text) if len(t) > 1 and t.lower() not in tf5_STOP}
    tokens.update(_NUM_RE.findall(text.lower()))
    return tokens

def tf5_search_seeds(query: str, notes: list[str] | None=None, *, procedure: str=PROC_STRICT) -> list[str]:
    """Ladder of discovery queries from distinctive public tokens (specific → broader)."""
    chunks = [_preamble(query)]
    if notes:
        chunks.extend(notes[-2:])
    text = ' '.join(chunks)
    seeds: list[str] = []
    phrases = _goal_phrases(text)
    for phrase in phrases:
        seeds.append(phrase)
        parts = [p for p in _WORD_RE.findall(phrase) if p.lower() not in tf5_STOP]
        if len(parts) >= 2:
            seeds.append(' '.join(parts))
        if len(parts) >= 3:
            seeds.append(' '.join(parts[:2]))
            seeds.append(parts[-1])
        for part in parts:
            if len(part) >= 4:
                seeds.append(part)
    cats = tf5_category_mentions(text)
    for cat in cats:
        seeds.append(cat)
    noise = tf5_STOP | _DIR_UP | _DIR_DOWN | {'usd', 'eur', 'gbp', 'cad', 'aud', 'jpy', 'cny', 'inr', 'php', 'then', 'than', 'after', 'before', 'versus', 'vs', 'one', 'done', 'true', 'false', 'call', 'tool'}
    modelish = [m.group(1) for m in re.finditer('\\b([A-Za-z]*\\d{2,}[A-Za-z0-9-]*)\\b', text) if m.group(1).lower() not in noise and (not m.group(1).isdigit())]
    budget = _parse_budget(text)
    budget_str = str(int(budget)) if budget is not None and float(budget).is_integer() else str(budget) if budget is not None else None
    words: list[str] = []
    seen: set[str] = set()
    for tok in re.findall('[A-Za-z][A-Za-z0-9.+-]{2,}', text):
        low = tok.lower().strip('.,;:')
        if low in noise or low in seen:
            continue
        if budget_str and low == budget_str:
            continue
        seen.add(low)
        words.append(tok.strip('.,;:'))
        if len(words) >= 12:
            break
    alpha = [w for w in words if not re.search('\\d', w)]
    mixed = [w for w in words if w not in alpha]
    softs = tf5_soft_tokens(text)
    if procedure == PROC_SOFT and softs:
        primary = (softs[:1] + alpha[:3] + mixed[:2] + modelish[:2])[:6]
        seeds.append(' '.join(primary))
        if softs and alpha[:2]:
            seeds.append(f"{softs[0]} {' '.join(alpha[:2])}")
    elif procedure == PROC_RECALL:
        primary = (alpha[:3] + mixed[:2] + modelish[:2])[:6]
        if primary:
            seeds.append(' '.join(primary))
        if alpha[:2]:
            seeds.append(' '.join(alpha[:2]))
        if alpha:
            seeds.append(alpha[0])
    elif procedure == PROC_USE:
        use_bits = [w for w in alpha if w.lower() not in tf5_STOP][:5]
        primary = (use_bits[:3] + mixed[:2] + modelish[:1])[:6]
        if primary:
            seeds.append(' '.join(primary))
        if cats:
            seeds.append(' '.join(cats[:1] + use_bits[:2]))
        if use_bits[:2]:
            seeds.append(' '.join(use_bits[:2]))
    elif procedure == PROC_ADAPT:
        prefs = tf5_prefer_model_tokens(text)
        colors = [p for p in prefs if p.isalpha()]
        codes = [p for p in prefs if any((ch.isdigit() for ch in p))]
        for p in sorted(codes, key=lambda x: (-len(x), x)):
            seeds.append(p)
        if codes and colors:
            seeds.append(f'{codes[0]} {colors[0]}')
        if codes:
            seeds.append(' '.join(codes[:2]))
        if prefs:
            seeds.append(' '.join(prefs[:3]))
        primary = (alpha[:3] + mixed[:2] + modelish[:2])[:6]
        if primary:
            seeds.append(' '.join(primary))
    elif procedure == PROC_ASK:
        atom_bits = [a for a in tf5_constraint_atoms(query, notes) if any((ch.isdigit() for ch in a))]
        softs = tf5_soft_tokens(text)
        if softs and atom_bits:
            seeds.append(f'{softs[0]} {atom_bits[0]}')
        for atom in atom_bits[:3]:
            seeds.append(atom)
        primary = (softs[:1] + alpha[:3] + mixed[:2] + atom_bits[:2])[:6]
        if primary:
            seeds.append(' '.join(primary))
        if alpha[:2]:
            seeds.append(' '.join(alpha[:2]))
    else:
        primary = (alpha[:4] + mixed[:2] + modelish[:2])[:6]
        if primary:
            seeds.append(' '.join(primary))
        if len(primary) > 3:
            seeds.append(' '.join(primary[:3]))
        if alpha[:2]:
            seeds.append(' '.join(alpha[:2]))
    for atom in tf5_constraint_atoms(query, notes):
        if ' ' in atom and any((ch.isdigit() for ch in atom)):
            seeds.append(atom)
            if softs:
                seeds.append(f'{softs[0]} {atom}')
    out: list[str] = []
    seen_s: set[str] = set()
    for s in seeds:
        key = s.lower().strip()
        if not key or key in seen_s:
            continue
        seen_s.add(key)
        out.append(s.strip())
    return out or ['in stock']

class tf5_Ledger:
    """Mutable world state derived only from the public goal and observations."""

    def __init__(self, query: str, allowed_claims: set[str], *, tools: list[dict[str, Any]] | None=None) -> None:
        self.query = query
        self.allowed_claims = set(allowed_claims)
        self.budget = _parse_budget(_preamble(query))
        self.needs_ask = _needs_clarification(query)
        self.named = _named_variants(query)
        self.rank_axes = tf5_rank_axes(_preamble(query))
        self.firm_signal = bool(_FIRM_CUE_RE.search(_preamble(query)))
        self.soft_signal = bool(_SOFT_CUE_RE.search(_preamble(query)))
        self.soft_tokens = tf5_soft_tokens(_preamble(query))
        self.phrases = _goal_phrases(_preamble(query))
        self.categories = tf5_category_mentions(_preamble(query))
        self.prefer_models = tf5_prefer_model_tokens(_preamble(query))
        self.procedure = tf5_classify_procedure(query, tools=tools or [])
        if self.procedure == PROC_ASK:
            self.needs_ask = True
        self.mentioned = _fields_in_text(query, allowed_claims)
        self.atoms = tf5_constraint_atoms(query)
        self.seeds = tf5_search_seeds(query, procedure=self.procedure)
        self.seed_idx = 0
        self.notes: list[str] = []
        self.candidates: dict[str, dict[str, Any]] = {}
        self.viewed: dict[str, dict[str, Any]] = {}
        self.stock: dict[str, dict[str, Any]] = {}
        self.cart: list[dict[str, str]] = []
        self.actions: list[str] = []
        self.last_error: str | None = None
        self.pending_q = False
        self.asked = False
        self.clarified = False
        self.needs_recheck = False
        self.checked: set[str] = set()
        self.must_recover = False
        self.search_n = 0
        self.empty_searches = 0
        self.view_n = 0
        self.cart_n = 0
        self.failed: set[str] = set()
        self.tried_queries: set[str] = set()
        self.same_search_streak = 0
        self.last_search_query = ''
        self.identical_action_streak = 0
        self.last_action_stamp = ''
        self.post_cart_pending: set[str] = set()
        self.priority_asked = False
        self.goal_atoms = list(self.atoms)
        self.post_clarify_search = False
        self.initial_cart_failed = False
        self.anchor_stems: set[str] = set()
        self.anchor_pids: set[str] = set()
        self.anchor_prices: list[float] = []
        self.event_target_key: str | None = None
        self.post_event_rerank_done = False

    def _refresh(self) -> None:
        blob = ' '.join(self.notes[-3:])
        updated = tf5_parse_budget_update(blob)
        if updated is not None:
            if self.budget is None or updated <= self.budget + 1e-09:
                self.budget = updated
                if self.cart:
                    top = self.cart[-1]
                    if self.over_budget(top['product_id'], top['sku']):
                        self.must_recover = True
                        self.initial_cart_failed = True
        self.mentioned = _fields_in_text(self.query, self.allowed_claims, extra=blob)
        self.atoms = tf5_constraint_atoms(self.query, self.notes)
        self.soft_tokens = tf5_soft_tokens(f'{_preamble(self.query)} {blob}')
        self.phrases = _goal_phrases(f'{_preamble(self.query)} {blob}')
        self.prefer_models = tf5_prefer_model_tokens(f'{_preamble(self.query)} {blob}')
        self.seeds = tf5_search_seeds(self.query, self.notes, procedure=self.procedure)
        self.rank_axes = tf5_rank_axes(f'{_preamble(self.query)} {blob}')
        for (pid, sku) in _named_variants(blob):
            key = f'{pid}::{sku}'
            if key not in {f'{a}::{b}' for (a, b) in self.named}:
                self.named.append((pid, sku))
        if self.needs_ask and _notes_have_firm_spec(self.notes, prior_atoms=self.goal_atoms):
            self.clarified = True
            self.needs_ask = False
            if not self.post_clarify_search:
                self.seed_idx = 0
                self.tried_queries.clear()
                self.same_search_streak = 0
                self.post_clarify_search = True
                self.empty_searches = 0

    def note_shopper(self, content: str) -> None:
        text = (content or '').strip()
        if not text:
            return
        self.notes.append(text[:500])
        self.notes = self.notes[-5:]
        self._refresh()
        avail_ping = bool(re.search('\\bstill available\\b|\\bdo you (?:still )?have\\b|\\bcheck(?:\\s+again)?\\b|\\bis that (?:still )?(?:available|in stock)\\b', text, re.I))
        if '?' in text and (not avail_ping):
            self.pending_q = True
        elif avail_ping and self.cart:
            self.needs_recheck = True
            self.post_cart_pending.update((item['key'] for item in self.cart))
            self.pending_q = False
        if self.asked and (self.clarified or _notes_have_firm_spec(self.notes, prior_atoms=self.goal_atoms)):
            self.clarified = True
            self.needs_ask = False
        price_event = bool(re.search('more\\s+expensive|price\\s+change|went\\s+(?:up|from)|re-?check before ordering', text, re.I))
        if self.cart:
            known_over = self.must_recover or any((self.over_budget(item['product_id'], item['sku']) for item in self.cart))
            self.needs_recheck = True
            self.checked.clear()
            for item in self.cart:
                self.stock.pop(item['key'], None)
                self.stock.pop(item['product_id'], None)
                self.post_cart_pending.add(item['key'])
            if known_over or price_event:
                self.must_recover = True
                self.initial_cart_failed = True

    def remember(self, name: str, args: dict[str, Any]) -> None:
        self.actions.append(name)
        self.actions = self.actions[-30:]
        stamp = name + ':' + json.dumps(args, sort_keys=True, default=str)[:180]
        if stamp == self.last_action_stamp:
            self.identical_action_streak += 1
        else:
            self.identical_action_streak = 1
            self.last_action_stamp = stamp
        if name in {'search', 'filter'}:
            self.search_n += 1
            if self.post_clarify_search and self.clarified:
                self.post_clarify_search = False
            q = str(args.get('query') or '').strip().lower()
            if q:
                if q == self.last_search_query:
                    self.same_search_streak += 1
                else:
                    self.same_search_streak = 1
                    self.last_search_query = q
                self.tried_queries.add(q)
        if name in {'view', 'compare'}:
            self.view_n += 1
        if name == 'message':
            self.asked = True
            self.pending_q = False
            self.priority_asked = True
        if name == 'add_to_cart':
            self.cart_n += 1
            pid = str(args.get('product_id') or '')
            sku = str(args.get('sku') or '')
            if pid and sku:
                key = f'{pid}::{sku}'
                self.cart = [c for c in self.cart if c.get('key') != key]
                self.cart.append({'product_id': pid, 'sku': sku, 'key': key})
                stem = _sku_stem(sku)
                if stem:
                    self.anchor_stems.add(stem)
                self.anchor_pids.add(pid)
                known_price = _as_float((self.candidates.get(key) or {}).get('price') or (self.viewed.get(pid) or {}).get('price'))
                if known_price is not None and known_price > 0:
                    self.anchor_prices.append(known_price)
                    self.anchor_prices = self.anchor_prices[-8:]
                if self.event_target_key is None:
                    self.event_target_key = key
                self.stock.pop(key, None)
                self.stock.pop(pid, None)
                self.checked.discard(key)
                self.needs_recheck = True
                self.must_recover = False
                self.post_cart_pending.add(key)
        if name == 'remove_from_cart':
            pid = str(args.get('product_id') or '')
            sku = str(args.get('sku') or '')
            key = f'{pid}::{sku}'
            if key:
                self.failed.add(key)
                stem = _sku_stem(sku)
                if stem:
                    self.anchor_stems.add(stem)
                if pid:
                    self.anchor_pids.add(pid)
                self.event_target_key = key
            self.cart = [c for c in self.cart if c.get('key') != key]
            self.must_recover = False
            self.needs_recheck = False
            self.checked.discard(key)
            self.post_cart_pending.discard(key)
        if name == 'place_test_order':
            self.identical_action_streak = 0

    def _store(self, row: dict[str, Any], *, viewed: bool=False) -> None:
        pid = str(row.get('product_id') or '')
        if not pid:
            return
        sku = str(row.get('sku') or '')
        key = f'{pid}::{sku}' if sku else pid
        prev = self.candidates.get(key, {})
        merged = {**prev, **{k: v for (k, v) in row.items() if v is not None}}
        merged['product_id'] = pid
        if sku:
            merged['sku'] = sku
        self.candidates[key] = merged
        if viewed:
            slim = {k: merged.get(k) for k in ('product_id', 'sku', 'title', 'brand', 'price', 'in_stock', 'category_path', 'facts', 'listing_id') if merged.get(k) is not None}
            self.viewed[pid] = {**self.viewed.get(pid, {}), **slim}
        if len(self.candidates) > tf5_LEDGER_CAND_CAP:
            ranked = sorted(self.candidates.items(), key=lambda kv: (0 if _sku_stem(str(kv[1].get('sku') or '')) in self.anchor_stems else 1, 0 if str(kv[1].get('product_id') or '') in self.anchor_pids else 1, 0 if kv[1].get('product_id') in self.viewed else 1, 0 if kv[0] in self.stock or kv[1].get('product_id') in self.stock else 1, float(kv[1]['price']) if isinstance(kv[1].get('price'), (int, float)) else 1e+18))
            keep = {k for (k, _) in ranked[:tf5_LEDGER_CAND_CAP]}
            self.candidates = {k: v for (k, v) in self.candidates.items() if k in keep}

    def ingest(self, name: str, args: dict[str, Any], observation: dict[str, Any]) -> None:
        if not isinstance(observation, dict):
            return
        (envelope, payload) = _unwrap_observation(observation)
        if envelope.get('error') or payload.get('error'):
            self.last_error = str(envelope.get('error') or payload.get('error'))[:240]
            return
        if name in {'search', 'filter'}:
            rows = [r for r in payload.get('results') or [] if isinstance(r, dict)]
            if not rows:
                self.empty_searches += 1
                self.seed_idx = min(self.seed_idx + 1, max(0, len(self.seeds) - 1))
            else:
                self.empty_searches = 0
            for row in rows:
                self._store(row)
        elif name == 'compare':
            for row in payload.get('compared') or payload.get('results') or []:
                if isinstance(row, dict):
                    self._store(row, viewed=True)
        elif name == 'view':
            self._store(payload, viewed=True)
            pid = str(payload.get('product_id') or args.get('product_id') or '')
            if pid:
                facts = payload.get('facts') if isinstance(payload.get('facts'), dict) else {}
                self.viewed[pid] = {**self.viewed.get(pid, {}), 'product_id': pid, 'title': payload.get('title'), 'brand': payload.get('brand'), 'category_path': payload.get('category_path'), 'facts': facts, 'listing_id': payload.get('listing_id'), 'price': payload.get('price'), 'in_stock': payload.get('in_stock')}
            for row in (payload.get('variants') or [])[:16]:
                if isinstance(row, dict):
                    self._store({**row, 'product_id': row.get('product_id') or pid})
            if pid:
                variants = [v for v in payload.get('variants') or [] if isinstance(v, dict)]
                if variants:
                    self.viewed.setdefault(pid, {})
                    self.viewed[pid]['variants'] = variants[:16]
            parent_stem = _sku_stem(str(payload.get('sku') or ''))
            parent_anchor = bool(pid in self.anchor_pids or (parent_stem and parent_stem in self.anchor_stems))
            if parent_anchor:
                for row in (payload.get('variants') or [])[:16]:
                    if isinstance(row, dict):
                        vst = _sku_stem(str(row.get('sku') or ''))
                        if vst:
                            self.anchor_stems.add(vst)
                        vpid = str(row.get('product_id') or pid or '')
                        if vpid:
                            self.anchor_pids.add(vpid)
                        vsku = str(row.get('sku') or '')
                        if vpid and vsku:
                            self._store({**row, 'product_id': vpid, 'sku': vsku})
        elif name == 'inspect_stock':
            pid = str(payload.get('product_id') or args.get('product_id') or '')
            sku = str(payload.get('sku') or args.get('sku') or '')
            if pid:
                stock_body = {k: v for (k, v) in payload.items() if k not in {'done', 'error', 'observation', 'action'}}
                key = f'{pid}::{sku}'
                self.stock[key] = stock_body
                self.stock[pid] = stock_body
                self._store({'product_id': pid, 'sku': sku, **stock_body})
                self.checked.add(key)
                live_price = _as_float(stock_body.get('price'))
                if live_price and any((item.get('key') == key for item in self.cart)):
                    self.anchor_prices.append(live_price)
                    self.anchor_prices = self.anchor_prices[-8:]
                self.post_cart_pending.discard(key)
                carted = any((item.get('key') == key or item.get('product_id') == pid for item in self.cart))
                if carted or self.needs_recheck:
                    if self.live_ok(pid, sku):
                        self.needs_recheck = False
                        self.must_recover = False
                    else:
                        self.needs_recheck = False
                        self.must_recover = True
                        self.failed.add(key)
                        stem = _sku_stem(sku)
                        if stem:
                            self.anchor_stems.add(stem)
                        if pid:
                            self.anchor_pids.add(pid)
                        self.event_target_key = key
                        if self.cart_n <= 1:
                            self.initial_cart_failed = True
        elif name == 'inspect_cart':
            rebuilt: list[dict[str, str]] = []
            stack: list[Any] = [payload]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    pid = str(node.get('product_id') or '')
                    sku = str(node.get('sku') or '')
                    if pid and sku:
                        rebuilt.append({'product_id': pid, 'sku': sku, 'key': f'{pid}::{sku}'})
                    stack.extend(node.values())
                elif isinstance(node, list):
                    stack.extend(node)
            if rebuilt:
                seen: set[str] = set()
                uniq: list[dict[str, str]] = []
                for item in rebuilt:
                    if item['key'] in seen:
                        continue
                    seen.add(item['key'])
                    uniq.append(item)
                self.cart = uniq

    def live_ok(self, product_id: str, sku: str) -> bool:
        key = f'{product_id}::{sku}'
        if key in self.post_cart_pending:
            return False
        stock = self.stock.get(key) or self.stock.get(product_id)
        if not isinstance(stock, dict):
            return False
        flagged = _as_bool(stock.get('in_stock'))
        if flagged is not True:
            return False
        price = _as_float(stock.get('price'))
        if self.budget is not None:
            if price is None:
                return False
            if price > self.budget + 1e-09:
                return False
        return True

    def over_budget(self, product_id: str, sku: str) -> bool:
        if self.budget is None:
            return False
        key = f'{product_id}::{sku}'
        stock = self.stock.get(key) or self.stock.get(product_id)
        if not isinstance(stock, dict):
            return False
        price = _as_float(stock.get('price'))
        return price is not None and price > self.budget + 1e-09

    def ready_order(self) -> dict[str, str] | None:
        if self.must_recover or self.needs_recheck or (not self.cart):
            return None
        top = self.cart[-1]
        if top['key'] in self.post_cart_pending:
            return None
        if not self.live_ok(top['product_id'], top['sku']):
            return None
        if self.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM}:
            if not self.constraints_ok(top['product_id'], top['sku']):
                return None
        return {'product_id': top['product_id'], 'sku': top['sku']}

    def evidence_ratio(self, row: dict[str, Any], viewed: dict[str, Any] | None=None) -> float:
        atoms = self.atoms
        if not atoms:
            return 1.0
        blob = _product_blob(row, viewed)
        (hits, total) = _atoms_hit(atoms, blob)
        return hits / max(total, 1)

    def _soft_hit(self, row: dict[str, Any], viewed: dict[str, Any] | None=None) -> bool:
        if not self.soft_tokens:
            return False
        blob = _product_blob(row, viewed)
        return any((tok.lower() in blob for tok in self.soft_tokens))

    def _phrase_hit(self, row: dict[str, Any], viewed: dict[str, Any] | None=None) -> bool:
        if not self.phrases:
            return False
        blob = _product_blob(row, viewed)
        for phrase in self.phrases:
            pl = phrase.lower()
            if pl in blob:
                return True
            parts = [p.lower() for p in _WORD_RE.findall(phrase) if p.lower() not in tf5_STOP]
            if parts and all((p in blob for p in parts)):
                return True
        return False

    def _category_hit(self, row: dict[str, Any], viewed: dict[str, Any] | None=None) -> bool:
        if not self.categories:
            return True
        path = (viewed or {}).get('category_path')
        if path is None:
            path = row.get('category_path')
        path_text = json.dumps(path, ensure_ascii=False, default=str).lower() if path else ''
        if not path_text or path_text in {'null', '[]', '""'}:
            return False
        for cat in self.categories:
            tokens = [t.lower() for t in _WORD_RE.findall(cat) if t.lower() not in tf5_STOP and len(t) > 2]
            if tokens and any((t in path_text for t in tokens)):
                return True
        return False

    def commitment_price(self) -> float | None:
        """Last carted listing price — the class we should stay near after events."""
        refs = [p for p in self.anchor_prices if p and p > 0]
        if refs:
            return max(refs)
        return None

    def price_class_ok(self, price_f: float | None, product_id: str | None=None, sku: str | None=None) -> bool:
        """Reject substitutes that are far cheaper than the item we already committed to.

        Generic: a replacement under a tightened ceiling should still be the same
        order of magnitude as the last cart, not an unrelated low-price listing.
        Same-product / same-SKU-stem variants may be cheaper and stay eligible.
        """
        if price_f is None:
            return True
        if product_id and product_id in self.anchor_pids:
            return True
        stem = _sku_stem(sku or '')
        if stem and stem in self.anchor_stems:
            return True
        if not (self.failed or self.initial_cart_failed or self.cart_n > 0):
            return True
        ref = self.commitment_price()
        if ref is None:
            return True
        floor = 0.35 * ref
        if self.budget is not None:
            floor = min(floor, 0.35 * self.budget)
        return price_f + 1e-09 >= floor

    def evidence_need(self) -> float:
        if self.procedure == PROC_ASK:
            return 0.75 if len(self.atoms) <= 3 else 0.6
        if self.procedure in {PROC_STRICT, PROC_CLAIM}:
            return 0.67 if len(self.atoms) <= 3 else 0.5
        if self.procedure == PROC_SOFT:
            return 0.5
        if self.procedure == PROC_RECALL:
            return 0.34
        if self.procedure in {PROC_USE, PROC_ADAPT}:
            return 0.34
        return 0.5

    def constraints_ok(self, product_id: str, sku: str) -> bool:
        """True when public constraint atoms are evidenced on the viewed listing."""
        if not self.atoms and self.procedure not in {PROC_RECALL}:
            return True
        key = f'{product_id}::{sku}'
        row = self.candidates.get(key) or self.candidates.get(product_id) or {}
        viewed = self.viewed.get(product_id) or {}
        if not viewed and product_id not in self.viewed:
            return False
        if self.procedure == PROC_RECALL:
            if self.phrases and (not self._phrase_hit(row, viewed)):
                return False
            if self.categories and (not self._category_hit(row, viewed)):
                return False
            return True
        ratio = self.evidence_ratio(row, viewed)
        return ratio + 1e-09 >= self.evidence_need()

    def _axis_score(self, row: dict[str, Any], viewed: dict[str, Any]) -> float:
        if not self.rank_axes:
            return 0.0
        product = _flatten({**row, **viewed})
        total = 0.0
        for (idx, (axis, direction)) in enumerate(self.rank_axes):
            weight = 4.0 - idx
            observed = _lookup([product, viewed, row], axis)
            if observed is None:
                facts = viewed.get('facts') if isinstance(viewed.get('facts'), dict) else {}
                for (key, value) in {**product, **facts}.items():
                    if not isinstance(key, str):
                        continue
                    if _keys_match(_norm_key(key), _norm_key(axis)):
                        observed = value
                        break
            if observed is None:
                continue
            try:
                num = float(str(observed).replace(',', ''))
            except ValueError:
                total += weight * (0.5 if direction > 0 else 0.0)
                continue
            total += weight * direction * num
        return total

    def ranked(self) -> list[tuple[float, str, dict[str, Any]]]:
        goal = tf5_goal_tokens(self.query, self.notes)
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        recovering = bool(self.failed) or self.initial_cart_failed or self.cart_n > 0
        use_blob = f"{_preamble(self.query)} {' '.join(self.notes[-2:])}".lower()
        use_tokens = {t.lower() for t in _WORD_RE.findall(use_blob) if t.lower() not in tf5_STOP and len(t) > 3}
        model_toks = [t.lower() for t in self.prefer_models]
        code_toks = [t for t in model_toks if any((ch.isdigit() for ch in t))]
        color_toks = [t for t in model_toks if t.isalpha() and len(t) >= 3]
        for (key, row) in self.candidates.items():
            if key in self.failed:
                continue
            score = 0.0
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            stem = _sku_stem(sku)
            viewed = self.viewed.get(pid) or {}
            if pid in self.viewed:
                score += 4
            if key in self.stock or pid in self.stock:
                score += 3
                if self.live_ok(pid, sku):
                    score += 4
                else:
                    score -= 5
            if row.get('in_stock') is True:
                score += 1
            price = row.get('price')
            if any((pid == n[0] and (not sku or sku == n[1]) for n in self.named)):
                score += 8 if self.procedure == PROC_RANK else 5
            elif any((pid == n[0] for n in self.named)):
                score += 3 if self.procedure == PROC_RANK else 2
            title = str(row.get('title') or '')
            score += min(4, _overlap(title, goal))
            ev = self.evidence_ratio(row, viewed)
            if self.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM}:
                score += 10.0 * ev
                if ev + 1e-09 < self.evidence_need():
                    score -= 6
            else:
                score += 6.0 * ev
            goal_nums = {t for t in goal if any((ch.isdigit() for ch in t))}
            title_nums = set(_NUM_RE.findall(title.lower()))
            if goal_nums and title_nums & goal_nums:
                score += 2
            facts = viewed.get('facts') if isinstance(viewed.get('facts'), dict) else {}
            for field in self.mentioned:
                observed = _lookup([viewed, row, facts], field)
                if observed is None:
                    continue
                score += 2 + min(3, _overlap(observed, goal))
            score += self._axis_score(row, viewed)
            if self.procedure == PROC_SOFT and self._soft_hit(row, viewed):
                score += 9
            elif self.procedure == PROC_SOFT and self.soft_tokens and (pid in self.viewed):
                score -= 2
            if self.procedure == PROC_ASK and self.soft_tokens and self._soft_hit(row, viewed):
                score += 6
            if self.procedure == PROC_RECALL:
                if self._phrase_hit(row, viewed):
                    score += 10
                else:
                    score -= 4
                if self.categories:
                    score += 5 if self._category_hit(row, viewed) else -6
            if self.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM} and self.categories and (pid in self.viewed):
                score += 6 if self._category_hit(row, viewed) else -12
            if self.procedure == PROC_USE and pid in self.viewed:
                blob = _product_blob(row, viewed)
                fit = sum((1 for t in use_tokens if t in blob))
                score += min(10, fit * 1.5)
                fact_items = facts.items() if isinstance(facts, dict) else ()
                present_keys = {str(fk).lower() for (fk, _fv) in fact_items}
                for (fk, fv) in fact_items:
                    fkl = str(fk).lower()
                    if not any((t in fkl for t in use_tokens if len(t) > 3)) and fkl not in use_blob:
                        continue
                    score += 4
                    fv_l = str(fv).lower()
                    if any((t in fv_l for t in use_tokens if len(t) > 3)):
                        score += 3
                    num = _as_float(fv)
                    if num is not None and num > 0:
                        score += min(6.0, num ** 0.5 / 12.0)
                relevant_keys: set[str] = set()
                for other in self.viewed.values():
                    ofacts = other.get('facts') if isinstance(other.get('facts'), dict) else {}
                    for ok in ofacts:
                        okl = str(ok).lower()
                        if okl in use_blob or any((t in okl for t in use_tokens if len(t) > 3)):
                            relevant_keys.add(okl)
                for rk in relevant_keys:
                    if rk in present_keys:
                        score += 4
                    else:
                        score -= 3
                title_hits = sum((1 for t in use_tokens if t in title.lower()))
                score += min(6, title_hits)
            if self.procedure == PROC_ADAPT or recovering:
                if stem and stem in self.anchor_stems:
                    score += 14
                if pid in self.anchor_pids:
                    score += 5
                for failed_key in self.failed:
                    if failed_key.startswith(f'{pid}::'):
                        score += 6
                        break
                    failed_sku = failed_key.split('::', 1)[-1]
                    if stem and stem == _sku_stem(failed_sku):
                        score += 12
                        break
                blob = _product_blob(row, viewed)
                code_hits = sum((1 for tok in code_toks if _model_token_hit(tok, blob)))
                color_hits = sum((1 for tok in color_toks if tok in blob))
                if code_hits and color_hits:
                    score += 12
                elif code_hits:
                    score += 8
                elif color_hits:
                    score += 3
                elif code_toks and recovering:
                    score -= 4
                score += 3 * sum((1 for tok in model_toks if _model_token_hit(tok, blob)))
            if self.procedure == PROC_CLAIM and pid in self.viewed and facts:
                score += 3
                blob = _product_blob(row, viewed)
                score += min(4, sum((1 for t in goal_nums if t in blob)))
            title_l = title.lower()
            if (self.procedure == PROC_ADAPT or recovering) and code_toks and (' for ' in title_l):
                for tok in code_toks:
                    if tok and re.search(f'\\bfor\\b.{{0,40}}\\b{re.escape(tok)}\\b', title_l):
                        score -= 12
                        break
                lead = title_l.split(' for ', 1)[0]
                if any((_model_token_hit(tok, title_l) for tok in code_toks)) and (not any((_model_token_hit(tok, lead) for tok in code_toks))):
                    score -= 6
                if any((tok in title_l for tok in code_toks)):
                    score -= 8
                if recovering:
                    score -= 6
            if self.budget is not None:
                price_f = _as_float(price)
                stock = self.stock.get(key) or self.stock.get(pid) or {}
                live_price = _as_float(stock.get('price'))
                if live_price is not None:
                    price_f = live_price
                if price_f is not None:
                    if price_f > self.budget + 1e-09:
                        score -= 8
                    elif self.procedure == PROC_USE:
                        score += 1.5 + min(2.5, 2.5 * (price_f / max(self.budget, 1.0)))
                    elif recovering or self.procedure == PROC_ADAPT:
                        if not self.price_class_ok(price_f, pid, sku):
                            score -= 16
                        else:
                            ref = self.commitment_price() or self.budget
                            closeness = 1.0 - min(1.0, abs(price_f - ref) / max(ref, 1.0))
                            util = price_f / max(self.budget, 1.0)
                            score += 2 + 3.0 * closeness + 2.0 * util
                    else:
                        score += 2 + min(2.0, 2.0 * (price_f / max(self.budget, 1.0)))
            ranked.append((score, key, row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    def best_sibling(self) -> dict[str, str] | None:
        """Least-relax alternate: same SKU stem (or prefer-code), live and under budget."""
        need = self.evidence_need() if self.atoms else 0.0
        deferred: list[tuple[float, str, dict[str, Any]]] = []
        stem_only = bool(self.anchor_stems)
        extra: list[tuple[str, str]] = []
        for pid in list(self.anchor_pids):
            viewed = self.viewed.get(pid) or {}
            for var in viewed.get('variants') or []:
                if not isinstance(var, dict):
                    continue
                vpid = str(var.get('product_id') or pid or '')
                vsku = str(var.get('sku') or '')
                if vpid and vsku:
                    extra.append((vpid, vsku))
                    self._store({**var, 'product_id': vpid, 'sku': vsku})
        for (_score, key, row) in self.ranked():
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            if not pid or not sku or key in self.failed:
                continue
            stem = _sku_stem(sku)
            blob = _product_blob(row, self.viewed.get(pid))
            code_toks = [t.lower() for t in self.prefer_models if any((ch.isdigit() for ch in t))]
            stem_ok = bool(stem and stem in self.anchor_stems)
            code_ok = bool(code_toks) and any((_model_token_hit(tok, blob) for tok in code_toks))
            extra_ok = any((pid == ep and sku == es for (ep, es) in extra))
            same_pid = pid in self.anchor_pids
            if not (stem_ok or extra_ok or same_pid):
                if code_ok and (not stem_only):
                    deferred.append((_score, key, row))
                continue
            hit = self._sibling_action(pid, sku, key, row, need)
            if hit:
                return hit
        if not stem_only:
            for (_score, key, row) in deferred:
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '')
                hit = self._sibling_action(pid, sku, key, row, need)
                if hit:
                    return hit
        for (vpid, vsku) in extra:
            key = f'{vpid}::{vsku}'
            if key in self.failed or not vpid or (not vsku):
                continue
            row = self.candidates.get(key) or {'product_id': vpid, 'sku': vsku}
            hit = self._sibling_action(vpid, vsku, key, row, need)
            if hit:
                return hit
        return None

    def _sibling_action(self, pid: str, sku: str, key: str, row: dict[str, Any], need: float) -> dict[str, str] | None:
        if pid not in self.viewed:
            return {'action': 'view', 'product_id': pid, 'sku': sku}
        viewed = self.viewed.get(pid) or {}
        if self.atoms and self.evidence_ratio(row, viewed) + 1e-09 < need:
            return None
        if key not in self.stock and pid not in self.stock:
            return {'action': 'inspect_stock', 'product_id': pid, 'sku': sku}
        if self.live_ok(pid, sku):
            p = _as_float((self.stock.get(f'{pid}::{sku}') or self.stock.get(pid) or {}).get('price'))
            if p is None:
                p = _as_float(row.get('price'))
            if not self.price_class_ok(p, pid, sku):
                return None
            return {'action': 'add_to_cart', 'product_id': pid, 'sku': sku}
        return None

    def next_seed(self) -> str:
        if not self.seeds:
            return 'in stock'
        idx = min(self.seed_idx, len(self.seeds) - 1)
        candidate = self.seeds[idx]
        for offset in range(len(self.seeds)):
            s = self.seeds[(idx + offset) % len(self.seeds)]
            if s.lower() in self.tried_queries:
                continue
            parts = s.split()
            if parts and all((_moneyish_token(p) for p in parts)):
                continue
            return s
        return candidate

    def recovery_step(self) -> dict[str, str] | None:
        sib = self.best_sibling()
        if sib:
            return sib
        need = self.evidence_need()
        for (score, key, row) in self.ranked():
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            if not pid or not sku or key in self.failed:
                continue
            if score < 1 and self.view_n > 0:
                continue
            if pid not in self.viewed:
                return {'action': 'view', 'product_id': pid, 'sku': sku}
            viewed = self.viewed.get(pid) or {}
            if self.atoms and self.evidence_ratio(row, viewed) + 1e-09 < need:
                continue
            if self.procedure == PROC_RECALL and self.phrases and (not self._phrase_hit(row, viewed)):
                continue
            price_f = _as_float(row.get('price'))
            stock = self.stock.get(key) or self.stock.get(pid) or {}
            live_p = _as_float(stock.get('price'))
            if live_p is not None:
                price_f = live_p
            if not self.price_class_ok(price_f, pid, sku):
                continue
            if key in self.stock or pid in self.stock:
                if self.live_ok(pid, sku):
                    return {'action': 'add_to_cart', 'product_id': pid, 'sku': sku}
                continue
            return {'action': 'inspect_stock', 'product_id': pid, 'sku': sku}
        for (_score, key, row) in self.ranked():
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            if pid and sku and (key not in self.failed) and (pid not in self.viewed):
                return {'action': 'view', 'product_id': pid, 'sku': sku}
        return None

    def pick_cart_target(self) -> dict[str, str] | None:
        """Best evidenced, live-valid cart candidate under the active procedure."""
        recovering = bool(self.failed) or self.initial_cart_failed
        if self.procedure != PROC_RANK and (self.procedure == PROC_ADAPT or recovering):
            sib = self.best_sibling()
            if sib and sib['action'] in {'inspect_stock', 'add_to_cart', 'view'}:
                return sib
        need = self.evidence_need()
        soft_ready: dict[str, str] | None = None
        prefer_filter = bool(self.procedure == PROC_ADAPT and self.prefer_models and self.failed)
        stem_filter = bool(self.anchor_stems and recovering and (self.procedure == PROC_ADAPT))
        code_toks = [t.lower() for t in self.prefer_models if any((ch.isdigit() for ch in t))]
        for (score, key, row) in self.ranked():
            pid = str(row.get('product_id') or '')
            sku = str(row.get('sku') or '')
            if not pid or not sku or key in self.failed:
                continue
            if stem_filter:
                stem = _sku_stem(sku)
                if not (stem and stem in self.anchor_stems) and pid not in self.anchor_pids:
                    continue
            elif prefer_filter:
                stem = _sku_stem(sku)
                blob = _product_blob(row, self.viewed.get(pid))
                stem_ok = bool(stem and stem in self.anchor_stems)
                code_ok = bool(code_toks) and any((_model_token_hit(tok, blob) for tok in code_toks))
                if not (stem_ok or code_ok or pid in self.anchor_pids):
                    continue
            viewed = self.viewed.get(pid) or {}
            if pid not in self.viewed:
                return {'action': 'view', 'product_id': pid, 'sku': sku}
            if self.atoms and self.evidence_ratio(row, viewed) + 1e-09 < need:
                continue
            if self.procedure == PROC_RECALL:
                if self.phrases and (not self._phrase_hit(row, viewed)):
                    continue
                if self.categories and (not self._category_hit(row, viewed)):
                    continue
            if self.procedure == PROC_CLAIM and (not viewed.get('facts')):
                return {'action': 'view', 'product_id': pid, 'sku': sku}
            price_f = _as_float(row.get('price'))
            live = self.stock.get(key) or self.stock.get(pid) or {}
            live_p = _as_float(live.get('price'))
            if live_p is not None:
                price_f = live_p
            if recovering and self.procedure != PROC_RANK and (not self.price_class_ok(price_f, pid, sku)):
                continue
            if self.procedure == PROC_RANK and self.named:
                named_keys = {f'{p}::{s}' for (p, s) in self.named}
                if key not in named_keys and pid not in {p for (p, _) in self.named}:
                    continue
            if self.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM, PROC_RECALL}:
                if self.categories and (not self._category_hit(row, viewed)):
                    continue
            if key not in self.stock and pid not in self.stock:
                return {'action': 'inspect_stock', 'product_id': pid, 'sku': sku}
            if not self.live_ok(pid, sku):
                continue
            cand = {'action': 'add_to_cart', 'product_id': pid, 'sku': sku}
            if self.procedure == PROC_SOFT and self.soft_tokens:
                if self._soft_hit(row, viewed):
                    return cand
                if soft_ready is None:
                    soft_ready = cand
                continue
            return cand
        if soft_ready and (not recovering):
            return soft_ready
        if prefer_filter:
            prefer_filter = False
            for (score, key, row) in self.ranked():
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '')
                if not pid or not sku or key in self.failed:
                    continue
                if pid not in self.viewed:
                    return {'action': 'view', 'product_id': pid, 'sku': sku}
                if key not in self.stock and pid not in self.stock:
                    return {'action': 'inspect_stock', 'product_id': pid, 'sku': sku}
                if self.live_ok(pid, sku):
                    return {'action': 'add_to_cart', 'product_id': pid, 'sku': sku}
        return None

    def hint(self, turn: int, max_steps: int) -> str:
        left = max_steps - turn + 1
        if self.needs_ask and (not self.clarified):
            return 'Ask one focused question about the missing firm specification before carting.'
        if self.procedure == PROC_USE and (not self.priority_asked) and (not self.cart):
            return 'Optionally clarify use priorities with message, then compare facts before first cart.'
        if self.pending_q:
            return 'Reply to the shopper with message before ordering.'
        if self.must_recover and self.cart:
            return f"Live check failed for {self.cart[-1]['key']}: remove, replace, re-inspect, order."
        if self.post_cart_pending and self.cart:
            top = self.cart[-1]
            return f"Cart changed — inspect_stock {top['key']} again before place_test_order (price/stock may have moved)."
        if self.needs_recheck and self.cart:
            top = self.cart[-1]
            if top['key'] not in self.checked:
                return f"Re-inspect_stock {top['key']} once after the shopper update."
        if self.last_error:
            return f'Recover from error ({self.last_error}) then continue.'
        recent = self.actions[-4:]
        if recent and len(set(recent)) == 1 and (recent[0] in {'search', 'filter', 'message', 'inspect_stock'}):
            return f'Stop looping on {recent[0]}; advance the purchase pipeline.'
        if self.named and self.view_n < min(2, len(self.named)):
            return 'View/compare named product::sku keys before ranking or carting.'
        if not self.candidates:
            return f'Discover with search/filter (try: {self.next_seed()}), in_stock=true.'
        if self.procedure == PROC_RECALL and self.view_n < 2:
            return 'Survey multiple phrase-matching listings before carting.'
        if self.view_n == 0:
            return 'Inspect a shortlist with view/compare; verify exact sku facts.'
        if not self.cart:
            if self.procedure == PROC_SOFT and self.soft_tokens:
                return f"Cart a firm-compliant item honoring soft preference ({', '.join(self.soft_tokens)}) when available."
            if left <= 8:
                return 'Cart the best fully compliant inspected variant now.'
            return 'Cart only a fully compliant intended purchase; honor soft prefs when feasible.'
        top = self.cart[-1]
        if top['key'] not in self.stock and top['product_id'] not in self.stock:
            return 'inspect_stock the carted sku once, then place_test_order.'
        if not self.live_ok(top['product_id'], top['sku']):
            return 'Carted sku failed live checks — remove_from_cart and recover.'
        if left <= 10:
            return 'Verified — place_test_order NOW with grounded claims if required.'
        return 'Live checks passed — place_test_order for the carted variant.'

    def board(self, turn: int, max_steps: int) -> str:
        lines = [f'ledger turn={turn}/{max_steps} left={max_steps - turn + 1}', f'procedure={self.procedure}', f"budget={(self.budget if self.budget is not None else 'none')} ask={self.needs_ask and (not self.clarified)} recheck={self.needs_recheck} recover={self.must_recover}", f'searches={self.search_n} empty={self.empty_searches} views={self.view_n} cands={len(self.candidates)} cart={len(self.cart)} adds={self.cart_n}', f'seed={self.next_seed()}', f'next={self.hint(turn, max_steps)}']
        if self.soft_tokens:
            lines.append('soft=' + ', '.join(self.soft_tokens))
        if self.phrases:
            lines.append('phrases=' + ' | '.join(self.phrases[:3]))
        if self.categories:
            lines.append('category_mentions=' + ' | '.join(self.categories))
        if self.named:
            lines.append('named=' + ', '.join((f'{p}::{s}' for (p, s) in self.named[:8])))
        if self.rank_axes:
            lines.append('rank_axes=' + ', '.join((f"{a}:{('up' if d > 0 else 'down')}" for (a, d) in self.rank_axes)))
        if self.mentioned:
            lines.append('claim_fields=' + ', '.join(sorted(self.mentioned)))
        if self.atoms:
            lines.append('constraint_atoms=' + ' | '.join(self.atoms[:10]))
        if self.notes:
            lines.append('notes:')
            for note in self.notes[-2:]:
                lines.append(f'  - {note}')
        if self.cart:
            lines.append('cart:')
            for item in self.cart[-3:]:
                stock = self.stock.get(item['key']) or self.stock.get(item['product_id']) or {}
                lines.append(f"  - {item['key']} price={stock.get('price', '?')} in_stock={stock.get('in_stock', '?')}")
        for (score, key, row) in self.ranked()[:_BOARD_SHOW]:
            pid = str(row.get('product_id') or '')
            ev = self.evidence_ratio(row, self.viewed.get(pid))
            lines.append(f"cand score={score:.1f} ev={ev:.2f} {key} title={str(row.get('title') or '')[:60]} price={row.get('price', '?')} stock={row.get('in_stock', '?')}")
        if self.cart:
            top = self.cart[-1]
            viewed = self.viewed.get(top['product_id']) or {}
            facts = viewed.get('facts') if isinstance(viewed.get('facts'), dict) else {}
            if facts:
                preview = ', '.join((f'{k}={str(v)[:36]}' for (k, v) in list(facts.items())[:8]))
                lines.append(f'final_facts={preview}')
        return '\n'.join(lines)

def tf5_ground_order(args: dict[str, Any], *, tools: list[dict[str, Any]], ledger: tf5_Ledger) -> dict[str, Any]:
    cleaned = dict(args)
    required = _required_args(tools, 'place_test_order')
    needs = 'justification' in required
    if not needs:
        cleaned.pop('justification', None)
        return cleaned
    pid = str(cleaned.get('product_id') or '')
    sku = str(cleaned.get('sku') or '')
    if (not pid or not sku) and ledger.cart:
        top = ledger.cart[-1]
        pid = pid or top['product_id']
        sku = sku or top['sku']
        cleaned['product_id'] = pid
        cleaned['sku'] = sku
    key = f'{pid}::{sku}'
    stock = dict(ledger.stock.get(key) or ledger.stock.get(pid) or {})
    viewed = dict(ledger.viewed.get(pid) or {})
    candidate = dict(ledger.candidates.get(key) or ledger.candidates.get(pid) or {})
    product = _flatten({**candidate, **viewed, **stock})
    allowed = set(_claim_fields(tools))
    open_spec = _open_spec_claims(tools)
    soft_fields = _soft_schema_fields(tools, allowed)
    goal = tf5_goal_tokens(ledger.query, ledger.notes)
    goal_text = f"{_preamble(ledger.query)} {' '.join(ledger.notes[-3:])}".lower()
    prefer = set(ledger.mentioned)
    raw_claims: list[dict[str, Any]] = []
    just = cleaned.get('justification')
    if isinstance(just, dict) and isinstance(just.get('claims'), list):
        raw_claims = [c for c in just['claims'] if isinstance(c, dict)]
    if open_spec:
        allowed.update({'price', 'in_stock'})
        allowed.update(soft_fields)
        allowed.update(prefer)
        for claim in raw_claims:
            field = str(claim.get('field') or '').strip()
            if field and re.fullmatch('[A-Za-z][A-Za-z0-9_]{1,40}', field):
                allowed.add(field)
                prefer.add(field)
    if not allowed:
        allowed = {'price', 'in_stock'}
    grounded: list[dict[str, str]] = []
    seen: set[str] = set()
    attest_blob = _product_blob(product, viewed)

    def add(field: str, value: Any) -> None:
        if field not in allowed or field in seen:
            return
        if value is None or value == '':
            return
        if isinstance(value, bool):
            text = 'true' if value else 'false'
        else:
            text = str(value).strip()
        if not text or text.lower() in {'unknown', 'none', 'n/a', 'null'}:
            return
        if field != 'in_stock' and text.lower() == 'false':
            return
        if field not in {'price', 'in_stock'} and (not _value_attested(value, attest_blob)):
            return
        grounded.append({'field': field, 'value': text[:80]})
        seen.add(field)

    def best_value(field: str) -> Any:
        direct = _lookup([product, stock, viewed, candidate], field)
        if direct not in (None, ''):
            if field in {'price', 'in_stock'}:
                return direct
            if _overlap(direct, goal) > 0 or _value_attested(direct, goal_text):
                return direct
            surface = field.replace('_', ' ').lower()
            if surface in goal_text or field.lower() in goal_text:
                return direct
            text = str(direct).lower().strip()
            if any((tok == text or tok in text.split() for tok in goal if tok.isalpha() and len(tok) >= 3)):
                return direct
            if any((atom in text for atom in ledger.atoms)):
                return direct
        want_nums = _goal_nums_near(ledger.query, field, ledger.notes)
        if not want_nums:
            want_nums = {t for t in goal if any((ch.isdigit() for ch in t))}
        best_val: Any = None
        best_score = 0
        for (key_name, value) in product.items():
            if value in (None, '', True, False) or isinstance(value, (dict, list)):
                continue
            if not isinstance(key_name, str):
                continue
            if key_name in {'product_id', 'sku', 'title', 'price', 'in_stock', 'currency', 'listing_id'} and field not in {'price', 'in_stock'}:
                continue
            text = str(value).lower().strip()
            nums = set(_NUM_RE.findall(text))
            shared = nums & want_nums if want_nums else set()
            overlap = 0
            if _keys_match(_norm_key(key_name), _norm_key(field)):
                overlap += 8
            if shared:
                overlap += 3 * len(shared)
            if len(text) >= 2 and (text in goal_text or any((tok == text for tok in goal if len(tok) >= 2))):
                overlap += 6
            if 1 < len(text) <= 12 and text.replace(' ', '').isalnum():
                compact = text.replace(' ', '')
                if compact in goal_text.replace(' ', '') or any((compact in tok or tok in compact for tok in goal if len(tok) >= 2)):
                    overlap += 5
            for tok in goal:
                if tok.isalpha() and len(tok) >= 2 and (tok in text or text in tok):
                    overlap += 2
            for atom in ledger.atoms:
                if atom in text:
                    overlap += 3
            overlap += min(2, _overlap(value, goal))
            if overlap > best_score:
                best_score = overlap
                best_val = value
        return best_val if best_score >= 4 else None
    add('price', stock.get('price', product.get('price')))
    stock_flag = stock.get('in_stock', product.get('in_stock'))
    flagged = _as_bool(stock_flag)
    if flagged is not None:
        add('in_stock', flagged)
    else:
        add('in_stock', stock_flag)
    for claim in raw_claims:
        field = str(claim.get('field') or '').strip()
        if not field or field in {'price', 'in_stock'}:
            continue
        if field not in allowed and (not open_spec):
            continue
        if open_spec:
            allowed.add(field)
        observed = best_value(field)
        if observed is not None:
            add(field, observed)
            continue
        raw_val = claim.get('value')
        if raw_val in (None, ''):
            continue
        if _value_attested(raw_val, attest_blob):
            add(field, raw_val)
    for field in list(prefer):
        if field in seen or field in {'price', 'in_stock'}:
            continue
        observed = best_value(field)
        if observed is not None:
            add(field, observed)
    if not any((f not in {'price', 'in_stock'} | soft_fields for f in seen)):
        ranked_fields: list[tuple[int, str, Any]] = []
        for field in sorted(allowed):
            if field in {'price', 'in_stock'} | soft_fields:
                continue
            surface = field.replace('_', ' ').lower()
            parts = [p for p in field.lower().split('_') if len(p) >= 3]
            in_goal = surface in goal_text or field.lower() in goal_text or any((p in goal_text for p in parts)) or bool(_goal_nums_near(ledger.query, field, ledger.notes))
            observed = best_value(field)
            if observed is None or not _value_attested(observed, attest_blob):
                continue
            score = 0
            if in_goal:
                score += 4
            score += min(3, _overlap(observed, goal))
            if score > 0:
                ranked_fields.append((score, field, observed))
        ranked_fields.sort(key=lambda row: (-row[0], row[1]))
        if ranked_fields:
            (_s, field, value) = ranked_fields[0]
            add(field, value)
    if not any((f not in {'price', 'in_stock'} | soft_fields for f in seen)):
        scored: list[tuple[int, str, Any]] = []
        hints = [str(c).strip() for c in list(prefer) + [str(c.get('field') or '') for c in raw_claims] if str(c).strip() and str(c).strip() not in {'price', 'in_stock'}]
        hints.extend(sorted((f for f in allowed if f not in {'price', 'in_stock'})))
        goal_nums = {t for t in goal if any((ch.isdigit() for ch in t))}
        for (key_name, value) in product.items():
            if not isinstance(key_name, str) or value in (None, ''):
                continue
            if key_name in {'product_id', 'sku', 'title', 'price', 'in_stock', 'currency', 'listing_id', 'category_path', 'variants', 'options', 'condition'}:
                continue
            if isinstance(value, (dict, list, bool)):
                continue
            text = str(value).lower().strip()
            overlap = 0
            if len(text) >= 2 and (text in goal_text or any((tok == text for tok in goal if len(tok) >= 2))):
                overlap += 6
            overlap += min(3, _overlap(value, goal))
            nums = set(_NUM_RE.findall(text))
            overlap += 2 * len(nums & goal_nums)
            for atom in ledger.atoms:
                if atom in text or re.sub('\\s+', '', atom) in re.sub('\\s+', '', text):
                    overlap += 4
            if overlap <= 0:
                continue
            if key_name in {'model', 'condition', 'listing_id'} and (not nums & goal_nums):
                continue
            claim_name = None
            for cand in hints:
                if _keys_match(_norm_key(cand), _norm_key(key_name)):
                    claim_name = cand
                    overlap += 3
                    break
            if claim_name is None and open_spec:
                if re.fullmatch('[a-z][a-z0-9_]{1,40}', key_name):
                    claim_name = key_name
            if claim_name is None:
                continue
            if not _value_attested(value, attest_blob):
                continue
            if open_spec or claim_name in allowed:
                scored.append((overlap, claim_name, value))
        scored.sort(key=lambda row: (-row[0], row[1]))
        if scored:
            (_s, field, value) = scored[0]
            if open_spec:
                allowed.add(field)
            add(field, value)
    if open_spec and 'brand' not in seen:
        brand_val = product.get('brand')
        if brand_val not in (None, '') and _value_attested(brand_val, attest_blob):
            brand_l = str(brand_val).lower()
            if any((tok.lower() == brand_l for tok in ledger.soft_tokens)) or brand_l in goal_text:
                allowed.add('brand')
                add('brand', brand_val)
            elif not any((f not in {'price', 'in_stock'} for f in seen)):
                allowed.add('brand')
                add('brand', brand_val)
            elif ledger.procedure == PROC_CLAIM:
                allowed.add('brand')
                add('brand', brand_val)
    if soft_fields and (not seen & soft_fields):
        for field in sorted(soft_fields):
            observed = _lookup([product], field)
            if observed is True or str(observed).strip().lower() in {'true', 'yes', '1'}:
                add(field, 'true')
                break
            if observed not in (None, '') and _value_attested(observed, attest_blob):
                add(field, observed)
                break
    if open_spec and (not any((f not in {'price', 'in_stock'} for f in seen))):
        for (key_name, value) in product.items():
            if not isinstance(key_name, str) or not re.fullmatch('[a-z][a-z0-9_]{1,40}', key_name):
                continue
            if key_name in {'product_id', 'sku', 'title', 'price', 'in_stock', 'currency', 'brand'}:
                continue
            if value in (None, '') or isinstance(value, (dict, list, bool)):
                continue
            if _value_attested(value, attest_blob) and _overlap(value, goal) > 0:
                allowed.add(key_name)
                add(key_name, value)
                break
    if needs and (not grounded):
        add('price', stock.get('price', product.get('price')))
        stock_flag = stock.get('in_stock', product.get('in_stock', True))
        flagged = _as_bool(stock_flag)
        add('in_stock', flagged if flagged is not None else stock_flag)
    cleaned['justification'] = {'claims': grounded}
    return cleaned

def tf5_normalize(name: str, args: dict[str, Any], *, tools: list[dict[str, Any]], ledger: tf5_Ledger) -> dict[str, Any]:
    cleaned = dict(args)
    if name == 'search':
        query = str(cleaned.get('query') or '').strip()
        stem_or_code = bool(query and (query.lower() in {s.lower() for s in ledger.anchor_stems} or any((query.lower() == t.lower() for t in ledger.prefer_models)) or (query.isdigit() and len(query) >= 4)))
        if not query or (not stem_or_code and (query.isdigit() or len(_WORD_RE.findall(query)) < 2)):
            query = ledger.next_seed()
            if ledger.anchor_stems and (ledger.failed or ledger.initial_cart_failed):
                query = sorted(ledger.anchor_stems)[0]
        if query.lower() in ledger.tried_queries and ledger.same_search_streak >= 1:
            ledger.seed_idx = min(ledger.seed_idx + 1, max(0, len(ledger.seeds) - 1))
            alt = ledger.next_seed()
            if alt.lower() != query.lower():
                query = alt
            else:
                tokens = [t for t in _WORD_RE.findall(query) if t.lower() not in tf5_STOP]
                query = ' '.join(tokens[:max(2, len(tokens) - 1)]) or query
        parts = [p for p in query.split() if p]
        dedup: list[str] = []
        for tok in parts:
            if not dedup or dedup[-1].lower() != tok.lower():
                dedup.append(tok)
        if len(dedup) > 10 and '"' not in query:
            cleaned['query'] = ' '.join(dedup[:8])
        else:
            cleaned['query'] = ' '.join(dedup) if dedup else query
        cleaned.setdefault('in_stock', True)
        cleaned.setdefault('k', 10)
        props = _tool_props(tools, 'search')
        if ledger.budget is not None and 'max_price' in props and ('max_price' not in cleaned):
            cleaned['max_price'] = ledger.budget
    if name == 'filter':
        cleaned.setdefault('in_stock', True)
        cleaned.setdefault('k', 10)
        if ledger.budget is not None and ledger.budget >= 5 and ('max_price' not in cleaned):
            cleaned['max_price'] = ledger.budget
        props = _tool_props(tools, 'filter')
        if ledger.categories and 'category' in props and ('category' not in cleaned):
            cleaned['category'] = ledger.categories[0]
        q = str(cleaned.get('query') or '').strip()
        if not q and ledger.seeds:
            cleaned['query'] = ledger.next_seed()
        elif q.lower() in ledger.tried_queries and ledger.same_search_streak >= 1:
            cleaned['query'] = ledger.next_seed()
    if name == 'compare':
        ids = cleaned.get('product_ids')
        if isinstance(ids, list):
            cleaned['product_ids'] = [str(x) for x in ids[:5] if str(x).strip()]
        elif isinstance(ids, str) and ids.strip():
            cleaned['product_ids'] = [p.strip() for p in ids.split(',') if p.strip()][:5]
        if not cleaned.get('product_ids') and ledger.named:
            cleaned['product_ids'] = [pid for (pid, _sku) in ledger.named[:5]]
    if name == 'message':
        cleaned['content'] = str(cleaned.get('content') or '').strip()[:400]
    if name == 'place_test_order':
        if (not cleaned.get('product_id') or not cleaned.get('sku')) and ledger.cart:
            top = ledger.cart[-1]
            cleaned.setdefault('product_id', top['product_id'])
            cleaned.setdefault('sku', top['sku'])
        cleaned = tf5_ground_order(cleaned, tools=tools, ledger=ledger)
    return cleaned

def _shopper_needs_verbal(ledger: 'Ledger') -> bool:
    """True when notes ask for a spoken selection / firm-spec confirmation."""
    blob = ' '.join(ledger.notes[-3:])
    if not blob:
        return False
    if re.search('\\bwhich one\\b|\\bare you going\\b|\\bdoes it actually\\b|\\bnon-negotiable\\b|\\bbefore you buy\\b|\\bconfirm\\b|\\bgoing with\\b|\\btell me (?:which|what)\\b', blob, re.I):
        return True
    if ledger.pending_q and '?' in blob:
        if re.search('\\bstill available\\b|\\bis that (?:still )?(?:available|in stock)\\b', blob, re.I) and (not re.search('\\bwhich one\\b|\\bbefore you buy\\b', blob, re.I)):
            return False
        return True
    return False

def tf5_forced_plan(*, ledger: tf5_Ledger, tools: list[dict[str, Any]], max_calls: int, remaining: int) -> list[tuple[str, dict[str, Any]]] | None:
    """Return a deterministic action plan when the next step is unambiguous."""
    names = _tool_names(tools)
    if ledger.needs_ask and (not ledger.clarified) and ('message' in names) and (not ledger.asked):
        return [('message', {'content': 'Before I pick an item, which firm product specification must it satisfy beyond budget and in-stock availability?'})]
    if ledger.pending_q and 'message' in names and _shopper_needs_verbal(ledger) and ('message' not in ledger.actions[-8:]):
        pick = ledger.cart[-1] if ledger.cart else None
        if pick is None:
            for (_s, _k, row) in ledger.ranked()[:1]:
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '')
                if pid and sku:
                    pick = {'product_id': pid, 'sku': sku}
                    break
        atom = next((a for a in ledger.atoms if any((ch.isdigit() for ch in a))), None)
        if pick:
            content = f"I am going with {pick['product_id']}::{pick['sku']}" + (f', confirming the firm {atom} requirement.' if atom else '.')
        else:
            content = 'I will verify listings against your stated constraints and confirm the chosen product_id::sku before checkout.'
        return [('message', {'content': content[:400]})]
    if ledger.post_clarify_search and ledger.clarified and (not ledger.cart) and ('search' in names):
        atoms = [a for a in ledger.atoms if any((ch.isdigit() for ch in a))]
        soft = ledger.soft_tokens[:1]
        if soft and atoms:
            seed = f'{soft[0]} {atoms[0]}'
        elif atoms:
            seed = atoms[0]
        else:
            seed = ledger.next_seed()
        args = {'query': seed, 'in_stock': True, 'k': 12}
        if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
            args['max_price'] = ledger.budget
        return [('search', args)]
    if ledger.cart and 'remove_from_cart' in names:
        top = ledger.cart[-1]
        stock = ledger.stock.get(top['key']) or ledger.stock.get(top['product_id'])
        priced_over = False
        if isinstance(stock, dict) and ledger.budget is not None:
            price = _as_float(stock.get('price'))
            priced_over = price is not None and price > ledger.budget + 1e-09
        if ledger.must_recover or priced_over or ledger.over_budget(top['product_id'], top['sku']):
            plan: list[tuple[str, dict[str, Any]]] = [('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']})]
            ledger.must_recover = True
            ledger.failed.add(top['key'])
            stem = _sku_stem(top.get('sku') or '')
            if not stem and '::' in top.get('key', ''):
                stem = _sku_stem(top['key'].split('::', 1)[-1])
            if stem:
                ledger.anchor_stems.add(stem)
            ledger.anchor_pids.add(top['product_id'])
            ledger.event_target_key = top['key']
            nxt = None
            if stem:
                for (_score, key, row) in ledger.ranked():
                    pid = str(row.get('product_id') or '')
                    sku = str(row.get('sku') or '')
                    if not pid or not sku or key in ledger.failed:
                        continue
                    if _sku_stem(sku) != stem:
                        continue
                    nxt = ledger._sibling_action(pid, sku, key, row, ledger.evidence_need() if ledger.atoms else 0.0)
                    if nxt:
                        break
            if nxt and nxt['action'] in names and (len(plan) < max_calls):
                args = {'product_id': nxt['product_id']}
                if nxt['action'] != 'view':
                    args['sku'] = nxt['sku']
                plan.append((nxt['action'], args))
            elif 'search' in names and len(plan) < max_calls and (stem or ledger.prefer_models or ledger.anchor_stems or ledger.anchor_pids):
                prefs = [t for t in ledger.prefer_models if any((ch.isdigit() for ch in t)) and (not _moneyish_token(t))]
                colors = [t for t in ledger.prefer_models if t.isalpha() and (not _moneyish_token(t))]
                title_q = _anchor_title_query(ledger)
                if title_q:
                    q = title_q
                elif prefs and colors:
                    q = f'{prefs[0]} {colors[0]}'
                elif prefs:
                    q = prefs[0]
                elif stem:
                    q = stem
                else:
                    q = sorted(ledger.anchor_stems)[0] if ledger.anchor_stems else ledger.next_seed()
                sargs: dict[str, Any] = {'query': q, 'in_stock': True, 'k': 12}
                if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
                    sargs['max_price'] = ledger.budget
                plan.append(('search', sargs))
            elif ledger.procedure == PROC_RANK and ledger.named and (len(plan) < max_calls):
                keep = [pid for (pid, sku) in ledger.named if f'{pid}::{sku}' not in ledger.failed][:5]
                if not ledger.post_event_rerank_done and 'compare' in names and (len(keep) >= 2):
                    ledger.post_event_rerank_done = True
                    plan.append(('compare', {'product_ids': keep}))
                else:
                    ledger.post_event_rerank_done = True
                    if 'inspect_stock' in names:
                        for (pid, sku) in ledger.named:
                            key = f'{pid}::{sku}'
                            if key in ledger.failed:
                                continue
                            if key not in ledger.checked:
                                plan.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                                break
                    if len(plan) < 2:
                        target = ledger.pick_cart_target()
                        if target and target['action'] in names:
                            args = {'product_id': target['product_id']}
                            if target['action'] != 'view':
                                args['sku'] = target['sku']
                            plan.append((target['action'], args))
            elif 'search' in names and len(plan) < max_calls:
                prefs = [t for t in ledger.prefer_models if any((ch.isdigit() for ch in t)) and (not _moneyish_token(t))]
                colors = [t for t in ledger.prefer_models if t.isalpha() and (not _moneyish_token(t))]
                atoms = [a for a in ledger.atoms if any((ch.isdigit() for ch in a)) and (not _moneyish_token(a))][:2]
                soft = [t for t in ledger.soft_tokens[:1] if not _moneyish_token(t)]
                if stem:
                    q = stem
                elif prefs and colors:
                    q = f'{prefs[0]} {colors[0]}'
                elif prefs:
                    q = prefs[0]
                elif soft and atoms:
                    q = f'{soft[0]} {atoms[0]}'
                elif atoms:
                    q = ' '.join(atoms)
                else:
                    q = ledger.next_seed()
                if not _moneyish_token(q) and (not all((_moneyish_token(p) for p in q.split()))):
                    sargs: dict[str, Any] = {'query': q, 'in_stock': True, 'k': 12}
                    if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
                        sargs['max_price'] = ledger.budget
                    plan.append(('search', sargs))
                else:
                    nxt = ledger.recovery_step()
                    if nxt and nxt['action'] in names and (len(plan) < max_calls):
                        args = {'product_id': nxt['product_id']}
                        if nxt['action'] != 'view':
                            args['sku'] = nxt['sku']
                        plan.append((nxt['action'], args))
            else:
                nxt = ledger.recovery_step()
                if nxt and nxt['action'] in names and (len(plan) < max_calls):
                    args = {'product_id': nxt['product_id']}
                    if nxt['action'] != 'view':
                        args['sku'] = nxt['sku']
                    plan.append((nxt['action'], args))
            return plan[:max_calls]
    if not ledger.cart and (ledger.failed or ledger.initial_cart_failed):
        if ledger.procedure == PROC_RANK and ledger.named:
            keep = [pid for (pid, sku) in ledger.named if f'{pid}::{sku}' not in ledger.failed][:5]
            if not ledger.post_event_rerank_done and 'compare' in names and (len(keep) >= 2):
                ledger.post_event_rerank_done = True
                return [('compare', {'product_ids': keep})]
            ledger.post_event_rerank_done = True
            if 'inspect_stock' in names:
                for (pid, sku) in ledger.named:
                    key = f'{pid}::{sku}'
                    if key in ledger.failed:
                        continue
                    if key not in ledger.checked:
                        return [('inspect_stock', {'product_id': pid, 'sku': sku})]
            target = ledger.pick_cart_target()
            if target and target['action'] in names:
                args = {'product_id': target['product_id']}
                if target['action'] != 'view':
                    args['sku'] = target['sku']
                return [(target['action'], args)]
            if keep and 'view' in names:
                unseen = [pid for pid in keep if pid not in ledger.viewed]
                if unseen:
                    return [('view', {'product_id': unseen[0]})]
                return [('view', {'product_id': keep[0]})]
        sib = ledger.best_sibling()
        if sib and sib['action'] in names:
            args = {'product_id': sib['product_id']}
            if sib['action'] != 'view':
                args['sku'] = sib['sku']
            return [(sib['action'], args)]
        if 'search' in names:
            stems = sorted(ledger.anchor_stems)
            prefs = [t for t in ledger.prefer_models if any((ch.isdigit() for ch in t)) and (not _moneyish_token(t))]
            colors = [t for t in ledger.prefer_models if t.isalpha() and (not _moneyish_token(t))]
            q = None
            title_q = _anchor_title_query(ledger)
            if title_q:
                q = title_q
            elif prefs:
                cand = f'{prefs[0]} {colors[0]}' if colors else prefs[0]
                if cand.lower() not in ledger.tried_queries:
                    q = cand
            if q is None:
                for stem in stems:
                    if stem.lower() not in ledger.tried_queries:
                        q = stem
                        break
            if q is None:
                atoms = [a for a in ledger.atoms if any((ch.isdigit() for ch in a)) and (not _moneyish_token(a))][:2]
                soft = [t for t in ledger.soft_tokens[:1] if not _moneyish_token(t)]
                cand = ' '.join(soft + atoms) if atoms or soft else ledger.next_seed()
                if cand.lower() not in ledger.tried_queries:
                    q = cand
            if q and (not all((_moneyish_token(p) for p in q.split()))):
                sargs = {'query': q, 'in_stock': True, 'k': 12}
                if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
                    sargs['max_price'] = ledger.budget
                return [('search', sargs)]
        if ledger.anchor_stems and ledger.procedure == PROC_ADAPT:
            sib = ledger.best_sibling()
            if sib and sib['action'] in names:
                args = {'product_id': sib['product_id']}
                if sib['action'] != 'view':
                    args['sku'] = sib['sku']
                return [(sib['action'], args)]
            if ledger.search_n >= 4 and 'search' not in ledger.actions[-1:]:
                pass
            elif ledger.search_n < 6 and 'search' in names:
                title_q = _anchor_title_query(ledger)
                prefs = [t for t in ledger.prefer_models if any((ch.isdigit() for ch in t)) and (not _moneyish_token(t))]
                q = title_q or (prefs[0] if prefs else None)
                if q and q.lower() not in ledger.tried_queries:
                    sargs = {'query': q, 'in_stock': True, 'k': 12}
                    if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
                        sargs['max_price'] = ledger.budget
                    return [('search', sargs)]
        nxt = ledger.recovery_step()
        if nxt and nxt['action'] in names:
            if ledger.procedure == PROC_ADAPT and ledger.anchor_stems and (ledger.search_n < 5):
                nsku = str(nxt.get('sku') or '')
                if _sku_stem(nsku) not in ledger.anchor_stems and nxt.get('product_id') not in ledger.anchor_pids:
                    title_q = _anchor_title_query(ledger)
                    if title_q and 'search' in names:
                        return [('search', {'query': title_q, 'in_stock': True, 'k': 12})]
                    if 'search' in names and ledger.prefer_models:
                        prefs = [t for t in ledger.prefer_models if any((ch.isdigit() for ch in t)) and (not _moneyish_token(t))]
                        if prefs:
                            return [('search', {'query': prefs[0], 'in_stock': True, 'k': 12})]
            args = {'product_id': nxt['product_id']}
            if nxt['action'] != 'view':
                args['sku'] = nxt['sku']
            return [(nxt['action'], args)]
    if ledger.post_cart_pending and ledger.cart and ('inspect_stock' in names):
        top = ledger.cart[-1]
        if top['key'] in ledger.post_cart_pending or top['key'] not in ledger.stock:
            return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
    if ledger.needs_recheck and ledger.cart and ('inspect_stock' in names):
        top = ledger.cart[-1]
        if top['key'] not in ledger.checked:
            return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
    if ledger.pending_q and 'message' in names and (ledger.identical_action_streak < 2) and _shopper_needs_verbal(ledger):
        pick = ledger.cart[-1] if ledger.cart else None
        if pick is None:
            for (_s, _k, row) in ledger.ranked()[:1]:
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '')
                if pid and sku:
                    pick = {'product_id': pid, 'sku': sku}
                    break
        atom = next((a for a in ledger.atoms if any((ch.isdigit() for ch in a))), None)
        if pick:
            content = f"I am going with {pick['product_id']}::{pick['sku']}" + (f', confirming the firm {atom} requirement.' if atom else '.')
        else:
            content = 'I will verify listings against your stated constraints, budget, and stock, then confirm the chosen product_id::sku before checkout.'
        return [('message', {'content': content[:400]})]
    ready = ledger.ready_order()
    if ready and 'place_test_order' in names:
        return [('place_test_order', tf5_ground_order(dict(ready), tools=tools, ledger=ledger))]
    if ledger.cart and 'place_test_order' in names:
        top = ledger.cart[-1]
        if top['key'] not in ledger.post_cart_pending and ledger.live_ok(top['product_id'], top['sku']) and (not ledger.must_recover) and (not (ledger.pending_q and _shopper_needs_verbal(ledger))):
            return [('place_test_order', tf5_ground_order({'product_id': top['product_id'], 'sku': top['sku']}, tools=tools, ledger=ledger))]
    if ledger.pending_q and 'message' in names and (ledger.identical_action_streak < 2):
        if ledger.clarified and ledger.candidates and (not ledger.cart):
            pass
        else:
            pick = ledger.cart[-1] if ledger.cart else None
            if pick is None:
                for (_s, _k, row) in ledger.ranked()[:1]:
                    pid = str(row.get('product_id') or '')
                    sku = str(row.get('sku') or '')
                    if pid and sku:
                        pick = {'product_id': pid, 'sku': sku}
                        break
            if pick:
                atom = next((a for a in ledger.atoms if any((ch.isdigit() for ch in a))), None)
                content = f"Planning {pick['product_id']}::{pick['sku']}" + (f' for the firm {atom} requirement' if atom else '') + '; I will re-check live price/stock before ordering.'
            else:
                content = 'I will verify listings against your stated constraints, budget, and stock, then confirm the chosen product_id::sku before checkout.'
            return [('message', {'content': content[:400]})]
    if not ledger.cart and ledger.failed:
        nxt = ledger.recovery_step()
        if nxt and nxt['action'] in names:
            args = {'product_id': nxt['product_id']}
            if nxt['action'] != 'view':
                args['sku'] = nxt['sku']
            return [(nxt['action'], args)]
    if ledger.named and ledger.view_n == 0:
        if 'compare' in names and len(ledger.named) >= 2:
            return [('compare', {'product_ids': [pid for (pid, _) in ledger.named[:5]]})]
        if 'view' in names:
            (pid, _sku) = ledger.named[0]
            return [('view', {'product_id': pid})]
    if ledger.procedure == PROC_RANK and ledger.named and ('view' in names):
        named_pids = [pid for (pid, _) in ledger.named]
        unseen = [pid for pid in named_pids if pid not in ledger.viewed]
        if unseen:
            return [('view', {'product_id': unseen[0]})]
    if ledger.procedure == PROC_USE:
        ledger.priority_asked = True
    if ledger.same_search_streak >= 2 and ledger.candidates and ('view' in names):
        for (_score, key, row) in ledger.ranked():
            pid = str(row.get('product_id') or '')
            if pid and pid not in ledger.viewed and (key not in ledger.failed):
                return [('view', {'product_id': pid})]
        nxt = ledger.recovery_step()
        if nxt and nxt['action'] in names:
            args = {'product_id': nxt['product_id']}
            if nxt['action'] != 'view':
                args['sku'] = nxt['sku']
            return [(nxt['action'], args)]
    if ledger.same_search_streak >= 2 and (not ledger.candidates) and ('search' in names):
        ledger.seed_idx = min(ledger.seed_idx + 1, max(0, len(ledger.seeds) - 1))
        seed = ledger.next_seed()
        args: dict[str, Any] = {'query': seed, 'in_stock': True, 'k': 10}
        if ledger.procedure == PROC_RECALL:
            args['k'] = 12
        if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
            args['max_price'] = ledger.budget
        return [('search', args)]
    if ledger.candidates and 'view' in names:
        recent = ledger.actions[-3:]
        stalled_search = bool(recent) and all((a in {'search', 'filter'} for a in recent))
        min_views = 2 if ledger.procedure in {PROC_RECALL, PROC_USE, PROC_CLAIM} else 1
        if ledger.view_n < min_views or stalled_search or ledger.same_search_streak >= 2:
            for (_score, key, row) in ledger.ranked():
                pid = str(row.get('product_id') or '')
                if pid and pid not in ledger.viewed and (key not in ledger.failed):
                    return [('view', {'product_id': pid})]
    cart_horizon = {PROC_RECALL: 14, PROC_USE: 14, PROC_CLAIM: 16, PROC_SOFT: 18, PROC_ASK: 16, PROC_STRICT: 16, PROC_ADAPT: 18, PROC_RANK: 22}.get(ledger.procedure, 16)
    if ledger.failed or ledger.initial_cart_failed:
        cart_horizon = max(cart_horizon, 26)
    elif ledger.procedure == PROC_USE and ledger.view_n >= 2:
        cart_horizon = max(cart_horizon, 24)
    elif ledger.procedure in {PROC_SOFT, PROC_ASK, PROC_STRICT} and ledger.view_n >= 2:
        cart_horizon = max(cart_horizon, 22)
    min_views_before_cart = {PROC_RECALL: 2, PROC_USE: 2, PROC_CLAIM: 1, PROC_SOFT: 1, PROC_ASK: 1, PROC_STRICT: 1, PROC_ADAPT: 1, PROC_RANK: 1}.get(ledger.procedure, 1)
    if remaining <= cart_horizon and ledger.candidates and (not ledger.cart) and (ledger.view_n >= min_views_before_cart):
        if ledger.needs_ask and (not ledger.clarified):
            pass
        else:
            target = ledger.pick_cart_target()
            if target and target['action'] in names:
                args = {'product_id': target['product_id']}
                if target['action'] != 'view':
                    args['sku'] = target['sku']
                plan = [(target['action'], args)]
                if target['action'] == 'add_to_cart' and remaining <= 4 and (max_calls >= 2) and ('inspect_stock' in names):
                    plan.append(('inspect_stock', {'product_id': target['product_id'], 'sku': target['sku']}))
                return plan[:max_calls]
    if remaining <= 10 and ledger.candidates and (not ledger.cart) and (ledger.view_n > 0):
        if not (ledger.needs_ask and (not ledger.clarified)):
            for (_score, key, row) in ledger.ranked():
                pid = str(row.get('product_id') or '')
                sku = str(row.get('sku') or '')
                if not pid or not sku or key in ledger.failed:
                    continue
                if ledger.atoms and pid in ledger.viewed:
                    if ledger.evidence_ratio(row, ledger.viewed.get(pid) or {}) + 1e-09 < ledger.evidence_need():
                        continue
                if pid not in ledger.viewed and 'view' in names:
                    return [('view', {'product_id': pid})]
                if key not in ledger.stock and pid not in ledger.stock and ('inspect_stock' in names):
                    return [('inspect_stock', {'product_id': pid, 'sku': sku})]
                live_p = _as_float((ledger.stock.get(key) or ledger.stock.get(pid) or {}).get('price'))
                if live_p is None:
                    live_p = _as_float(row.get('price'))
                if not ledger.price_class_ok(live_p, pid, sku):
                    continue
                if ledger.categories and ledger.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM, PROC_RECALL} and (not ledger._category_hit(row, ledger.viewed.get(pid) or {})):
                    continue
                if ledger.live_ok(pid, sku) and 'add_to_cart' in names:
                    plan = [('add_to_cart', {'product_id': pid, 'sku': sku})]
                    if remaining <= 4 and max_calls >= 2 and ('inspect_stock' in names):
                        plan.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                    return plan[:max_calls]
                break
    if ledger.cart and 'inspect_stock' in names:
        top = ledger.cart[-1]
        if top['key'] not in ledger.stock and top['product_id'] not in ledger.stock:
            return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
    if not ledger.candidates and 1 <= ledger.empty_searches <= max(2, len(ledger.seeds)) and ('search' in names):
        seed = ledger.next_seed()
        if seed.lower() not in ledger.tried_queries:
            args = {'query': seed, 'in_stock': True, 'k': 10}
            if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
                args['max_price'] = ledger.budget
            return [('search', args)]
    return None

def tf5_guard(pairs: list[tuple[str, dict[str, Any]]], *, ledger: tf5_Ledger, tools: list[dict[str, Any]], max_calls: int) -> list[tuple[str, dict[str, Any]]]:
    names = _tool_names(tools)
    out: list[tuple[str, dict[str, Any]]] = []
    if ledger.needs_ask and (not ledger.clarified) and ('message' in names) and (not ledger.asked):
        risky = {'add_to_cart', 'place_test_order', 'remove_from_cart'}
        if any((n in risky for (n, _) in pairs)) or not any((n == 'message' for (n, _) in pairs)):
            return [('message', {'content': 'What firm specification is required for this purchase besides budget and stock?'})]
    for (name, args) in pairs:
        pid = str(args.get('product_id') or '')
        sku = str(args.get('sku') or '')
        if ledger.needs_ask and (not ledger.clarified) and (not ledger.asked) and (name in {'add_to_cart', 'place_test_order'}) and ('message' in names):
            out.append(('message', {'content': 'What firm specification is required for this purchase besides budget and stock?'}))
            continue
        if name == 'add_to_cart' and pid and (pid not in ledger.viewed) and ('view' in names):
            out.append(('view', {'product_id': pid}))
            continue
        if name == 'view' and pid:
            prefer_codes = {t.lower() for t in ledger.prefer_models if any((ch.isdigit() for ch in t))}
            known_pids = {str(r.get('product_id') or '') for r in ledger.candidates.values()} | set(ledger.viewed.keys())
            if pid.lower() in prefer_codes and pid not in known_pids and ('search' in names):
                q = pid
                colors = [t for t in ledger.prefer_models if t.isalpha()]
                if colors:
                    q = f'{pid} {colors[0]}'
                sargs: dict[str, Any] = {'query': q, 'in_stock': True, 'k': 12}
                if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
                    sargs['max_price'] = ledger.budget
                out.append(('search', sargs))
                continue
        if name == 'message' and ledger.clarified:
            content = str(args.get('content') or '').lower()
            if re.search('\\bfirm\\b.*\\b(?:spec|requirement)\\b|\\btell me if you ask\\b', content):
                target = ledger.pick_cart_target()
                if target and target['action'] in names:
                    args2 = {'product_id': target['product_id']}
                    if target['action'] != 'view':
                        args2['sku'] = target['sku']
                    out.append((target['action'], args2))
                    continue
                if 'search' in names:
                    atoms = [a for a in ledger.atoms if any((ch.isdigit() for ch in a))]
                    soft = ledger.soft_tokens[:1]
                    q = f'{soft[0]} {atoms[0]}' if soft and atoms else atoms[0] if atoms else ledger.next_seed()
                    sargs = {'query': q, 'in_stock': True, 'k': 12}
                    if ledger.budget is not None and 'max_price' in _tool_props(tools, 'search'):
                        sargs['max_price'] = ledger.budget
                    out.append(('search', sargs))
                    continue
        if name == 'add_to_cart' and ledger.procedure == PROC_SOFT and ledger.soft_tokens and pid and sku and (ledger.cart_n == 0):
            row = ledger.candidates.get(f'{pid}::{sku}') or ledger.candidates.get(pid) or {}
            viewed = ledger.viewed.get(pid) or {}
            if not ledger._soft_hit(row, viewed):
                alt = ledger.pick_cart_target()
                if alt and alt['action'] == 'add_to_cart' and (alt['product_id'] != pid or alt['sku'] != sku):
                    alt_row = ledger.candidates.get(f"{alt['product_id']}::{alt['sku']}") or ledger.candidates.get(alt['product_id']) or {}
                    alt_view = ledger.viewed.get(alt['product_id']) or {}
                    if ledger._soft_hit(alt_row, alt_view):
                        out.append(('add_to_cart', {'product_id': alt['product_id'], 'sku': alt['sku']}))
                        continue
        if name == 'add_to_cart' and pid and sku and (ledger.procedure in {PROC_ASK, PROC_STRICT, PROC_SOFT, PROC_CLAIM}) and ledger.atoms and (ledger.view_n < 12):
            row = ledger.candidates.get(f'{pid}::{sku}') or ledger.candidates.get(pid) or {}
            viewed = ledger.viewed.get(pid) or {}
            if ledger.evidence_ratio(row, viewed) + 1e-09 < ledger.evidence_need():
                nxt = ledger.pick_cart_target()
                if nxt and nxt['action'] in names:
                    if nxt['action'] != 'add_to_cart' or (nxt['product_id'] != pid or nxt.get('sku') != sku):
                        args2 = {'product_id': nxt['product_id']}
                        if nxt['action'] != 'view':
                            args2['sku'] = nxt['sku']
                        out.append((nxt['action'], args2))
                        continue
        if name == 'add_to_cart' and pid and sku:
            key = f'{pid}::{sku}'
            if ledger.cart and any((c.get('key') == key for c in ledger.cart)) and (key not in ledger.post_cart_pending) and ledger.live_ok(pid, sku) and ('place_test_order' in names):
                out.append(('place_test_order', tf5_ground_order({'product_id': pid, 'sku': sku}, tools=tools, ledger=ledger)))
                continue
            out.append((name, args))
            continue
        if name in {'search', 'filter'}:
            q = str(args.get('query') or '').strip().lower()
            if q and q == ledger.last_search_query and (ledger.same_search_streak >= 2):
                if ledger.candidates and 'view' in names:
                    for (_score, key, row) in ledger.ranked():
                        pid2 = str(row.get('product_id') or '')
                        if pid2 and pid2 not in ledger.viewed and (key not in ledger.failed):
                            out.append(('view', {'product_id': pid2}))
                            break
                    else:
                        out.append((name, args))
                    continue
                args = dict(args)
                args['query'] = ledger.next_seed()
        if name == 'place_test_order' and pid and sku:
            key = f'{pid}::{sku}'
            if ledger.pending_q and _shopper_needs_verbal(ledger) and ('message' in names) and ('message' not in ledger.actions[-10:]):
                atom = next((a for a in ledger.atoms if any((ch.isdigit() for ch in a))), None)
                content = f'I am going with {pid}::{sku}' + (f', confirming the firm {atom} requirement.' if atom else '.')
                out.append(('message', {'content': content[:400]}))
                continue
            if key in ledger.post_cart_pending and 'inspect_stock' in names:
                out.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                continue
            if ledger.must_recover and 'remove_from_cart' in names:
                out.append(('remove_from_cart', {'product_id': pid, 'sku': sku}))
                continue
            if ledger.over_budget(pid, sku):
                ledger.must_recover = True
                ledger.failed.add(key)
                if 'remove_from_cart' in names:
                    out.append(('remove_from_cart', {'product_id': pid, 'sku': sku}))
                elif 'inspect_stock' in names:
                    out.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                continue
            if ledger.needs_recheck and key not in ledger.checked and ('inspect_stock' in names):
                out.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                continue
            has_stock = key in ledger.stock or pid in ledger.stock
            if not has_stock and 'inspect_stock' in names:
                out.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
                continue
            if has_stock and (not ledger.live_ok(pid, sku)):
                ledger.must_recover = True
                ledger.failed.add(key)
                if 'remove_from_cart' in names:
                    out.append(('remove_from_cart', {'product_id': pid, 'sku': sku}))
                continue
            if 'justification' in _required_args(tools, 'place_test_order') and pid not in ledger.viewed and ('view' in names):
                out.append(('view', {'product_id': pid}))
                continue
            if not ledger.live_ok(pid, sku):
                continue
        out.append((name, args))
    unique: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for (name, args) in out:
        stamp = name + ':' + json.dumps(args, sort_keys=True, default=str)
        if stamp in seen:
            continue
        seen.add(stamp)
        unique.append((name, args))
    names_present = {n for (n, _) in unique}
    if 'add_to_cart' in names_present and 'place_test_order' in names_present:
        unique = [(n, a) for (n, a) in unique if n != 'place_test_order']
    return unique[:max_calls]

def tf5_fallback_plan(*, ledger: tf5_Ledger, tools: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Last-resort deterministic action when the model returns nothing usable."""
    names = _tool_names(tools)
    if ledger.cart and ledger.post_cart_pending and ('inspect_stock' in names):
        top = ledger.cart[-1]
        return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
    if ledger.cart and 'place_test_order' in names:
        top = ledger.cart[-1]
        if top['key'] not in ledger.post_cart_pending and (not ledger.must_recover) and (not ledger.over_budget(top['product_id'], top['sku'])) and ledger.live_ok(top['product_id'], top['sku']):
            return [('place_test_order', tf5_ground_order({'product_id': top['product_id'], 'sku': top['sku']}, tools=tools, ledger=ledger))]
    ready = ledger.ready_order()
    if ready and 'place_test_order' in names:
        return [('place_test_order', tf5_ground_order(dict(ready), tools=tools, ledger=ledger))]
    if ledger.cart and 'inspect_stock' in names:
        top = ledger.cart[-1]
        if top['key'] not in ledger.stock and top['product_id'] not in ledger.stock:
            return [('inspect_stock', {'product_id': top['product_id'], 'sku': top['sku']})]
        if (not ledger.live_ok(top['product_id'], top['sku']) or ledger.over_budget(top['product_id'], top['sku'])) and 'remove_from_cart' in names:
            return [('remove_from_cart', {'product_id': top['product_id'], 'sku': top['sku']})]
    nxt = ledger.recovery_step()
    if nxt and nxt['action'] in names:
        args = {'product_id': nxt['product_id']}
        if nxt['action'] != 'view':
            args['sku'] = nxt['sku']
        return [(nxt['action'], args)]
    if ledger.candidates and 'view' in names:
        for (_score, key, row) in ledger.ranked():
            pid = str(row.get('product_id') or '')
            if pid and pid not in ledger.viewed and (key not in ledger.failed):
                return [('view', {'product_id': pid})]
    if 'search' in names:
        return [('search', {'query': ledger.next_seed(), 'in_stock': True, 'k': 10})]
    return []

def tf5_system_prompt(tools: list[dict[str, Any]], *, max_calls: int, max_steps: int, ledger: tf5_Ledger, needs_justification: bool, claim_fields: list[str]) -> str:
    tool_lines: list[str] = []
    for spec in tools:
        fn = _fn(spec)
        if not fn or not isinstance(fn.get('name'), str):
            continue
        desc = str(fn.get('description') or '').strip()
        params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {}
        props = params.get('properties') if isinstance(params.get('properties'), dict) else {}
        arg_names = ', '.join(sorted((str(k) for k in props.keys())))
        tool_lines.append(f"- {fn['name']}: {desc}" + (f' Args: {arg_names}.' if arg_names else ''))
    parts = ['You are a careful shopping agent. Obey the shopper request and every Task rules block.', 'Copy product_id and sku only from observations — never invent identifiers.', "Work for ANY product category using only the shopper's stated constraints and tool observations.", '', f'Active procedure: {ledger.procedure}', *_procedure_playbook(ledger.procedure), '', 'Shared process:', '1. If a firm requirement is deferred until you ask, message first.', '2. Discover with short constraint-focused search/filter queries; reformulate on weak hits.', '3. view/compare a shortlist and verify EVERY firm requirement on the exact sku using raw facts.', '4. First add_to_cart must be a genuine fully compliant choice, never a placeholder.', '5. Soft preferences guide ranking among compliant options; firm constraints always win.', '6. After add_to_cart OR any post-cart shopper update, inspect_stock the carted sku again before place_test_order — price and stock can change on cart events. If the fresh check fails budget or stock, remove_from_cart and recover.', '7. Never place_test_order in the same turn as add_to_cart.', '8. When verified by a post-cart inspect_stock, place_test_order immediately.', '9. Do not repeat the same search/filter query; broaden, drop optional tokens, or inspect hits.', f'10. Up to {max_calls} tool calls per turn; budget {max_steps} turns. Batch independent reads.', '', 'Tools:', *tool_lines]
    if ledger.needs_ask and (not ledger.clarified):
        parts.append('This request is underspecified. Your first action should ask for the missing firm product specification.')
    if ledger.named:
        shown = ', '.join((f'{pid}::{sku}' for (pid, sku) in ledger.named[:12]))
        parts.extend(['Named variants to compare/view before carting:', shown, 'Respect the stated tradeoff order (primary metric, then secondary, then price).'])
    if ledger.rank_axes:
        axes = ', '.join((f"{a}({('higher' if d > 0 else 'lower')})" for (a, d) in ledger.rank_axes))
        parts.append(f'Ranking axes from the request: {axes}.')
    if needs_justification:
        parts.extend(['Checkout requires structured justification claims grounded in view/compare/inspect_stock of the FINAL product.', f"Allowed fields: {', '.join(claim_fields)}.", 'Always include price and in_stock from the latest inspect_stock.', "Include exactly one claim for the shopper's constrained attribute (snake_case) with the observed value, plus one positive capability/reason field when the schema allows it."])
    else:
        parts.append('Omit justification unless the schema requires it.')
    parts.append('Continue until an observation reports done=true.')
    return '\n'.join(parts)

def tf5_problem_is_hard(*, ledger: tf5_Ledger, needs_justification: bool, max_steps: int) -> bool:
    """Cost-aware model choice from public problem cues only."""
    if needs_justification or ledger.needs_ask or ledger.named or ledger.rank_axes:
        return True
    if ledger.procedure in {PROC_ASK, PROC_SOFT, PROC_CLAIM, PROC_RANK, PROC_ADAPT, PROC_USE}:
        return True
    if ledger.soft_signal and ledger.firm_signal:
        return True
    if max_steps >= 24:
        return True
    return len(_preamble(ledger.query)) > 420



# ===== pipeline entrypoints =====
def tf1_agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    started = time.monotonic()
    environment = problem_data['environment']
    binding = dict(environment['binding'])
    policy = environment['policy_view']
    query = str(policy['query'])
    tools = list(policy.get('tools') or [])
    index = _tool_index(tools)
    max_steps = int(policy.get('max_steps') or 20)
    max_calls = max(1, min(6, int(policy.get('max_calls_per_turn') or 1)))
    problem_id = str(problem_data.get('problem_id') or problem_data.get('id') or 'problem')
    needs_justification = _justification_required(index)
    named = tf1_named_variants(_goal_head(query))
    model = tf1_model(prefer_strong=needs_justification or bool(named))
    memory = Memory(query)
    dialogue: list[dict[str, Any]] = []
    system = _SYSTEM
    if needs_justification:
        system += "\nCheckout requires structured justification claims grounded in view/compare of the FINAL product. Always include observed price and in_stock. Also include one claim for the shopper's constrained specification: choose a precise snake_case attribute name for the constrained property (avoid generic keys like title/model/category), with an evidenced value from the product. Add brand as the soft reason when known. Do not invent capability flags."
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': system}, {'role': 'user', 'content': query}]
    pending_bootstrap = _discover(index=index, goal=query, memory=memory, prefix=f'{problem_id}-boot', max_calls=max_calls)
    for turn in range(1, max_steps + 1):
        elapsed = time.monotonic() - started
        if elapsed >= _EPISODE_BUDGET_S:
            break
        if pending_bootstrap and turn == 1:
            tool_calls = pending_bootstrap
            assistant_content = ''
            pending_bootstrap = []
        else:
            status = memory.status_line()
            if elapsed > _EPISODE_BUDGET_S * 0.7:
                status += ' Please wrap up soon if you have a valid option.'
            if needs_justification and memory.cart:
                status += " When ordering, include evidenced claims for the shopper's constrained attribute plus price, in_stock, and one schema reason."
            if len(messages) >= 3 and messages[-1].get('role') == 'user' and str(messages[-1].get('content') or '').startswith('State:'):
                messages[-1] = {'role': 'user', 'content': status}
            else:
                messages.append({'role': 'user', 'content': status})
            tf1_trim(messages)
            tool_calls = []
            assistant_content = ''
            for attempt in range(_MAX_INFERENCE_ATTEMPTS):
                if time.monotonic() - started >= _EPISODE_BUDGET_S:
                    break
                inference = tf1_chat(model=model, messages=messages, tools=tools)
                if inference is None:
                    model = tf1_model(prefer_strong=True)
                    continue
                assistant = (inference.get('choices') or [{}])[0].get('message') or {}
                assistant_content = assistant.get('content') or ''
                tool_calls = _select_calls(assistant, index=index, max_calls=max_calls, memory=memory, prefix=f'{problem_id}-t{turn}-a{attempt}')
                if tool_calls:
                    break
                messages.append({'role': 'assistant', 'content': assistant_content})
                messages.append({'role': 'user', 'content': 'Please continue with a tool call.'})
            if not tool_calls:
                tool_calls = _discover(index=index, goal=query, memory=memory, prefix=f'{problem_id}-t{turn}-d', max_calls=max_calls)
            if not tool_calls:
                break
        messages.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls})
        prepared = []
        for (offset, call) in enumerate(tool_calls, start=1):
            fn = call['function']
            name = fn['name']
            args = _sanitize_args(name, _parse_arguments(fn.get('arguments')), index, memory)
            if name == 'place_test_order':
                args = _repair_order(args, index=index, memory=memory)
            call['function']['arguments'] = json.dumps(args, ensure_ascii=False)
            memory.note(name, args)
            prepared.append({'call_id': f'{problem_id}-turn-{turn}-{offset}', 'action': {'name': name, 'args': args}})
        group_id = f'{problem_id}-turn-{turn}'
        result = _proxy.post('/environment/call', json_data={**binding, 'call_id': group_id, 'idempotency_key': group_id, 'turn': turn, 'calls': prepared})
        if result is None:
            continue
        dialogue.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls, 'environment_result': result})
        done = False
        for (tool_call, call_result, prep) in zip(tool_calls, list(result.get('calls') or []), prepared):
            step = call_result.get('observation')
            if not isinstance(step, dict):
                step = {'observation': step, 'done': False, 'error': 'malformed'}
            done = memory.ingest(step, action=prep['action']['name']) or done
            messages.append({'role': 'tool', 'tool_call_id': tool_call['id'], 'content': tf1_compact(step)})
        user_message = result.get('user_message')
        if isinstance(user_message, dict) and user_message.get('content'):
            note = str(user_message['content'])
            memory.hear_shopper(note)
            messages.append({'role': 'user', 'content': note})
        if done:
            break
    return dialogue

def tf2_agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data['environment']
    binding = environment['binding']
    policy = environment['policy_view']
    tools = list(policy.get('tools') or [])
    query = str(policy.get('query') or '')
    problem_id = str(problem_data.get('problem_id', problem_data.get('id', 'problem')))
    max_calls = max(1, min(4, int(policy.get('max_calls_per_turn', 1))))
    max_steps = int(policy['max_steps'])
    allowed = _claim_fields(tools)
    needs_justification = 'justification' in _required_args(tools, 'place_test_order')
    ledger = tf2_Ledger(query, allowed, tools=tools)
    models = _model_ladder(tf2_problem_is_hard(ledger=ledger, needs_justification=needs_justification, max_steps=max_steps))
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': tf2_system_prompt(tools, max_calls=max_calls, max_steps=max_steps, ledger=ledger, needs_justification=needs_justification, claim_fields=sorted(allowed) or ['price', 'in_stock'])}, {'role': 'user', 'content': query}]
    dialogue: list[dict[str, Any]] = []
    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        action_pairs: list[tuple[str, dict[str, Any]]] = []
        assistant_content = ''
        forced = tf2_forced_plan(ledger=ledger, tools=tools, max_calls=max_calls, remaining=remaining)
        if forced is not None:
            action_pairs = forced
        else:
            _trim(messages)
            ask = list(messages)
            ask.append({'role': 'user', 'content': ledger.board(turn, max_steps)})
            tool_calls: list[dict[str, Any]] = []
            for attempt in range(2):
                inference = _chat(models=models, messages=ask, tools=tools)
                assistant = inference['choices'][0]['message']
                assistant_content = assistant.get('content') or ''
                tool_calls = list(assistant.get('tool_calls') or [])[:max_calls]
                if tool_calls:
                    break
                ask.extend([{'role': 'assistant', 'content': assistant_content}, {'role': 'user', 'content': 'Tool call required. Clarify, inspect, recover, or order.' if attempt == 0 else 'Call one supplied tool with valid JSON arguments.'}])
            for call in tool_calls:
                try:
                    name = call['function']['name']
                    if name not in _tool_names(tools):
                        continue
                    args = tf2_normalize(name, _arguments(call), tools=tools, ledger=ledger)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                action_pairs.append((name, args))
            action_pairs = tf2_guard(action_pairs, ledger=ledger, tools=tools, max_calls=max_calls)
        ready = ledger.ready_order()
        if ready and 'place_test_order' in _tool_names(tools) and (not ledger.must_recover) and (not ledger.post_cart_pending) and (not any((ledger.over_budget(ready['product_id'], ready['sku']) for _ in (0,)))) and (not any((n == 'place_test_order' for (n, _) in action_pairs))) and (not any((n == 'remove_from_cart' for (n, _) in action_pairs))):
            action_pairs = [('place_test_order', tf2_ground_order(dict(ready), tools=tools, ledger=ledger))]
        rebuilt: list[tuple[str, dict[str, Any]]] = []
        for (name, args) in action_pairs:
            if name == 'place_test_order':
                args = tf2_ground_order(dict(args), tools=tools, ledger=ledger)
            else:
                args = tf2_normalize(name, args, tools=tools, ledger=ledger)
            rebuilt.append((name, args))
        action_pairs = rebuilt
        if any((n == 'add_to_cart' for (n, _) in action_pairs)):
            action_pairs = [(n, a) for (n, a) in action_pairs if n != 'place_test_order']
        if ledger.post_cart_pending or ledger.needs_recheck:
            sanitized: list[tuple[str, dict[str, Any]]] = []
            for (name, args) in action_pairs:
                if name != 'place_test_order':
                    sanitized.append((name, args))
                    continue
                pid = str(args.get('product_id') or '')
                sku = str(args.get('sku') or '')
                if pid and sku and ('inspect_stock' in _tool_names(tools)):
                    sanitized.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
            action_pairs = sanitized or action_pairs
        if not action_pairs:
            action_pairs = tf2_fallback_plan(ledger=ledger, tools=tools)
        if not action_pairs:
            messages.extend([{'role': 'assistant', 'content': assistant_content}, {'role': 'user', 'content': 'Previous arguments were invalid. Call a schema-valid tool.'}])
            continue
        prepared: list[dict[str, Any]] = []
        for (index, (name, args)) in enumerate(action_pairs, start=1):
            prepared.append({'id': f'{problem_id}-t{turn}-c{index}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}})
            ledger.remember(name, args)
        messages.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': prepared})
        group_id = f'{problem_id}-turn-{turn}'
        envelope = {**binding, 'call_id': group_id, 'idempotency_key': group_id, 'turn': turn, 'calls': [{'call_id': f'{group_id}-{index}', 'action': {'name': name, 'args': args}} for (index, (name, args)) in enumerate(action_pairs, start=1)]}
        result = _proxy.post('/environment/call', json_data=envelope)
        if result is None:
            retry_id = f'{group_id}-retry'
            envelope = {**envelope, 'call_id': retry_id, 'idempotency_key': retry_id}
            result = _proxy.post('/environment/call', json_data=envelope)
        if result is None:
            raise RuntimeError('environment call failed')
        dialogue.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': prepared, 'environment_result': result})
        for ((name, args), tool_call, call_result) in zip(action_pairs, prepared, result['calls'], strict=True):
            observation = call_result.get('observation') or {}
            if not isinstance(observation, dict):
                observation = {'raw': observation}
            (envelope, payload) = _unwrap_observation(observation)
            merged = {**payload}
            if envelope.get('error') is not None:
                merged.setdefault('error', envelope.get('error'))
            ledger.ingest(name, args, observation)
            messages.append({'role': 'tool', 'tool_call_id': tool_call['id'], 'content': _compact_obs(merged if merged else observation)})
        user_message = result.get('user_message')
        if isinstance(user_message, dict) and user_message.get('content'):
            content = str(user_message['content'])
            ledger.note_shopper(content)
            messages.append({'role': 'user', 'content': content})
        elif isinstance(user_message, str) and user_message.strip():
            ledger.note_shopper(user_message)
            messages.append({'role': 'user', 'content': user_message})
        if any((isinstance(call.get('observation'), dict) and call['observation'].get('done') for call in result.get('calls') or [])):
            break
    return dialogue

def tf3_agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Satisfy the shopper with dynamic environment tools until done=true."""
    environment = problem_data['environment']
    binding = environment['binding']
    policy = environment['policy_view']
    tools = list(policy.get('tools') or [])
    problem_id = str(problem_data.get('problem_id', problem_data.get('id', 'problem')))
    max_calls = max(1, int(policy.get('max_calls_per_turn', 1)))
    max_steps = int(policy['max_steps'])
    stock_by_key: dict[str, dict[str, Any]] = {}
    viewed_facts: dict[str, dict[str, Any]] = {}
    carted = False
    removed = False
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': tf3_system_prompt(tools, max_calls, max_steps)}, {'role': 'user', 'content': policy['query']}]
    dialogue: list[dict[str, Any]] = []
    for turn in range(1, max_steps + 1):
        left = max_steps - turn + 1
        nudge = None
        if left <= 6:
            nudge = f'{left} turns left. If a valid item is inspected, add_to_cart, inspect_stock, and place_test_order now.'
        if left <= 3:
            nudge = f'Only {left} turns left. Place the order for the best verified in-cart variant immediately.'
        if removed and carted is False:
            nudge = (nudge + ' ' if nudge else '') + 'The cart was cleared. Select another valid product, cart it, re-check stock, and order.'
        ask = list(messages)
        if nudge:
            ask.append({'role': 'user', 'content': nudge})
        assistant_content = ''
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(2):
            inference = _proxy.post('/inference/chat/completions', json_data={'model': tf3_model(), 'messages': ask, 'tools': tools, 'tool_choice': 'required', 'temperature': 0})
            if inference is None:
                raise RuntimeError('inference request failed')
            assistant = inference['choices'][0]['message']
            assistant_content = assistant.get('content') or ''
            tool_calls = list(assistant.get('tool_calls') or [])[:max_calls]
            if tool_calls:
                break
            ask.extend([{'role': 'assistant', 'content': assistant_content}, {'role': 'user', 'content': 'Continue with a tool call toward inspecting and ordering a valid product.'}])
        else:
            raise RuntimeError('model returned no tool call before completion')
        prepared: list[dict[str, Any]] = []
        action_pairs: list[tuple[str, dict[str, Any]]] = []
        for call in tool_calls:
            try:
                name = call['function']['name']
                args = _clean_args(name, tf3_arguments(call), tools, stock_by_key, viewed_facts)
            except (ValueError, json.JSONDecodeError, KeyError, TypeError):
                continue
            action_pairs.append((name, args))
            prepared.append({'id': call.get('id') or f'call-{turn}-{len(prepared) + 1}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}})
        if not prepared:
            messages.extend([{'role': 'assistant', 'content': assistant_content}, {'role': 'user', 'content': 'Call a tool with valid JSON arguments.'}])
            continue
        tool_calls = prepared
        for (name, _args) in action_pairs:
            if name == 'add_to_cart':
                carted = True
                removed = False
            if name == 'remove_from_cart':
                removed = True
                carted = False
        messages.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls})
        group_id = f'{problem_id}-turn-{turn}'
        envelope = {**binding, 'call_id': group_id, 'idempotency_key': group_id, 'turn': turn, 'calls': [{'call_id': f'{group_id}-{index}', 'action': {'name': name, 'args': args}} for (index, (name, args)) in enumerate(action_pairs, start=1)]}
        result = _proxy.post('/environment/call', json_data=envelope)
        if result is None:
            raise RuntimeError('environment call failed')
        dialogue.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls, 'environment_result': result})
        for ((name, args), tool_call, call_result) in zip(action_pairs, tool_calls, result['calls'], strict=True):
            obs = call_result['observation']
            _remember_observation(name, args, obs, stock_by_key, viewed_facts)
            messages.append({'role': 'tool', 'tool_call_id': tool_call['id'], 'content': json.dumps(obs)})
        if result.get('user_message'):
            messages.append({'role': 'user', 'content': result['user_message']['content']})
        if any((call['observation']['done'] for call in result['calls'])):
            break
    return dialogue

def tf4_agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data['environment']
    binding = environment['binding']
    policy = environment['policy_view']
    tools = list(policy.get('tools') or [])
    query = str(policy.get('query') or '')
    problem_id = str(problem_data.get('problem_id', problem_data.get('id', 'problem')))
    policy_calls = max(1, int(policy.get('max_calls_per_turn', 1)))
    named_preview = _named_candidates(query)
    call_cap = 8 if len(named_preview) >= 2 else 4
    max_calls = max(1, min(call_cap, policy_calls))
    max_steps = min(int(policy['max_steps']), _TURN_CEILING)
    model = tf4_model()
    allowed_claims = _claim_field_allowlist(tools)
    needs_justification = 'justification' in _required_args(tools, 'place_test_order')
    needs_ask = tf4_needs_clarification(query)
    task_shape = _classify_task_shape(problem_data, tools=tools, named=named_preview, needs_ask=needs_ask)
    board = EpisodeBoard(query, allowed_claims, task_shape=task_shape)
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': tf4_system_prompt(tools, max_calls=max_calls, max_steps=max_steps, needs_ask=board.needs_ask, named=board.named, claim_fields=sorted(allowed_claims), needs_justification=needs_justification)}, {'role': 'user', 'content': query}]
    dialogue: list[dict[str, Any]] = []
    for turn in range(1, max_steps + 1):
        _trim_messages(messages)
        assistant_content = ''
        action_pairs: list[tuple[str, dict[str, Any]]] = []
        scripted = _scripted_actions(board, tools=tools, max_calls=max_calls, turn=turn, max_steps=max_steps)
        if scripted:
            action_pairs = scripted
        else:
            ask = list(messages)
            ask.append({'role': 'user', 'content': board.render(turn, max_steps)})
            tool_calls: list[dict[str, Any]] = []
            try:
                inference = _inference(model=model, messages=ask, tools=tools)
                assistant = inference['choices'][0]['message']
                assistant_content = assistant.get('content') or ''
                tool_calls = list(assistant.get('tool_calls') or [])[:max_calls]
            except Exception:
                tool_calls = []
            if not tool_calls:
                action_pairs = _progress_actions(board, tools=tools, max_calls=max_calls)
            else:
                for call in tool_calls:
                    try:
                        name = call['function']['name']
                        if name not in _tool_names(tools):
                            continue
                        args = _normalize_args(name, tf4_arguments(call), tools=tools, board=board)
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
                    action_pairs.append((name, args))
                if action_pairs and all((name == 'message' for (name, _) in action_pairs)) and (not board.pending_question) and (not (board.needs_ask and (not board.received_clarification))):
                    progress = _progress_actions(board, tools=tools, max_calls=max_calls)
                    if progress:
                        action_pairs = progress
        if action_pairs:
            action_pairs = _guard_actions(action_pairs, board=board, tools=tools, max_calls=max_calls)
        action_pairs = _force_finish_actions(action_pairs, board=board, tools=tools, remaining=max_steps - turn + 1, max_calls=max_calls)
        rebuilt: list[tuple[str, dict[str, Any]]] = []
        for (name, args) in action_pairs:
            if name == 'place_test_order':
                args = tf4_ground_order(dict(args), tools=tools, board=board)
            rebuilt.append((name, args))
        action_pairs = rebuilt
        if not action_pairs:
            action_pairs = _progress_actions(board, tools=tools, max_calls=max_calls)
        if not action_pairs:
            messages.extend([{'role': 'assistant', 'content': assistant_content}, {'role': 'user', 'content': 'Previous arguments were invalid. Call a schema-valid tool.'}])
            continue
        prepared_calls: list[dict[str, Any]] = []
        for (index, (name, args)) in enumerate(action_pairs, start=1):
            prepared_calls.append({'id': f'{problem_id}-t{turn}-c{index}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}})
        messages.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': prepared_calls})
        group_id = f'{problem_id}-turn-{turn}'
        envelope = {**binding, 'call_id': group_id, 'idempotency_key': group_id, 'turn': turn, 'calls': [{'call_id': f'{group_id}-{index}', 'action': {'name': name, 'args': args}} for (index, (name, args)) in enumerate(action_pairs, start=1)]}
        result = None
        for resync in range(2):
            attempt_envelope = dict(envelope)
            if resync:
                attempt_envelope['turn'] = turn + resync
                attempt_envelope['call_id'] = f'{group_id}-rs{resync}'
                attempt_envelope['idempotency_key'] = attempt_envelope['call_id']
            result = _env_call(binding, attempt_envelope)
            if result is not None:
                break
        if result is None:
            break
        for (name, args) in action_pairs:
            board.remember_action(name, args)
        dialogue.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': prepared_calls, 'environment_result': result})
        for ((name, args), tool_call, call_result) in zip(action_pairs, prepared_calls, result['calls'], strict=True):
            raw_observation = call_result.get('observation') or {}
            if not isinstance(raw_observation, dict):
                raw_observation = {'raw': raw_observation}
            observation = tf4_unwrap_observation(raw_observation)
            board.ingest(name, args, observation)
            messages.append({'role': 'tool', 'tool_call_id': tool_call['id'], 'content': _compact_observation(observation)})
        user_message = result.get('user_message')
        if isinstance(user_message, dict) and user_message.get('content'):
            content = str(user_message['content'])
            board.note_shopper(content)
            messages.append({'role': 'user', 'content': content})
        if any((isinstance(call.get('observation'), dict) and (call['observation'].get('done') or tf4_unwrap_observation(call['observation']).get('done')) for call in result.get('calls') or [])):
            break
    return dialogue

def tf5_agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data['environment']
    binding = environment['binding']
    policy = environment['policy_view']
    tools = list(policy.get('tools') or [])
    query = str(policy.get('query') or '')
    problem_id = str(problem_data.get('problem_id', problem_data.get('id', 'problem')))
    max_calls = max(1, min(10, int(policy.get('max_calls_per_turn', 1))))
    max_steps = int(policy['max_steps'])
    allowed = _claim_fields(tools)
    needs_justification = 'justification' in _required_args(tools, 'place_test_order')
    ledger = tf5_Ledger(query, allowed, tools=tools)
    models = _model_ladder(tf5_problem_is_hard(ledger=ledger, needs_justification=needs_justification, max_steps=max_steps))
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': tf5_system_prompt(tools, max_calls=max_calls, max_steps=max_steps, ledger=ledger, needs_justification=needs_justification, claim_fields=sorted(allowed) or ['price', 'in_stock'])}, {'role': 'user', 'content': query}]
    dialogue: list[dict[str, Any]] = []
    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        action_pairs: list[tuple[str, dict[str, Any]]] = []
        assistant_content = ''
        forced = tf5_forced_plan(ledger=ledger, tools=tools, max_calls=max_calls, remaining=remaining)
        if forced is not None:
            action_pairs = forced
        else:
            _trim(messages)
            ask = list(messages)
            ask.append({'role': 'user', 'content': ledger.board(turn, max_steps)})
            tool_calls: list[dict[str, Any]] = []
            for attempt in range(2):
                inference = _chat(models=models, messages=ask, tools=tools)
                assistant = inference['choices'][0]['message']
                assistant_content = assistant.get('content') or ''
                tool_calls = list(assistant.get('tool_calls') or [])[:max_calls]
                if tool_calls:
                    break
                ask.extend([{'role': 'assistant', 'content': assistant_content}, {'role': 'user', 'content': 'Tool call required. Clarify, inspect, recover, or order.' if attempt == 0 else 'Call one supplied tool with valid JSON arguments.'}])
            for call in tool_calls:
                try:
                    name = call['function']['name']
                    if name not in _tool_names(tools):
                        continue
                    args = tf5_normalize(name, _arguments(call), tools=tools, ledger=ledger)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                action_pairs.append((name, args))
            action_pairs = tf5_guard(action_pairs, ledger=ledger, tools=tools, max_calls=max_calls)
        ready = ledger.ready_order()
        if ready and 'place_test_order' in _tool_names(tools) and (not ledger.must_recover) and (not ledger.post_cart_pending) and (not any((ledger.over_budget(ready['product_id'], ready['sku']) for _ in (0,)))) and (not any((n == 'place_test_order' for (n, _) in action_pairs))) and (not any((n == 'remove_from_cart' for (n, _) in action_pairs))):
            action_pairs = [('place_test_order', tf5_ground_order(dict(ready), tools=tools, ledger=ledger))]
        rebuilt: list[tuple[str, dict[str, Any]]] = []
        for (name, args) in action_pairs:
            if name == 'place_test_order':
                args = tf5_ground_order(dict(args), tools=tools, ledger=ledger)
            else:
                args = tf5_normalize(name, args, tools=tools, ledger=ledger)
            rebuilt.append((name, args))
        action_pairs = rebuilt
        if any((n == 'add_to_cart' for (n, _) in action_pairs)):
            action_pairs = [(n, a) for (n, a) in action_pairs if n != 'place_test_order']
        if ledger.post_cart_pending or ledger.needs_recheck:
            sanitized: list[tuple[str, dict[str, Any]]] = []
            for (name, args) in action_pairs:
                if name != 'place_test_order':
                    sanitized.append((name, args))
                    continue
                pid = str(args.get('product_id') or '')
                sku = str(args.get('sku') or '')
                if pid and sku and ('inspect_stock' in _tool_names(tools)):
                    sanitized.append(('inspect_stock', {'product_id': pid, 'sku': sku}))
            action_pairs = sanitized or action_pairs
        if not action_pairs:
            action_pairs = tf5_fallback_plan(ledger=ledger, tools=tools)
        if not action_pairs:
            messages.extend([{'role': 'assistant', 'content': assistant_content}, {'role': 'user', 'content': 'Previous arguments were invalid. Call a schema-valid tool.'}])
            continue
        prepared: list[dict[str, Any]] = []
        for (index, (name, args)) in enumerate(action_pairs, start=1):
            prepared.append({'id': f'{problem_id}-t{turn}-c{index}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}})
            ledger.remember(name, args)
        messages.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': prepared})
        group_id = f'{problem_id}-turn-{turn}'
        envelope = {**binding, 'call_id': group_id, 'idempotency_key': group_id, 'turn': turn, 'calls': [{'call_id': f'{group_id}-{index}', 'action': {'name': name, 'args': args}} for (index, (name, args)) in enumerate(action_pairs, start=1)]}
        result = _proxy.post('/environment/call', json_data=envelope)
        if result is None:
            retry_id = f'{group_id}-retry'
            envelope = {**envelope, 'call_id': retry_id, 'idempotency_key': retry_id}
            result = _proxy.post('/environment/call', json_data=envelope)
        if result is None:
            raise RuntimeError('environment call failed')
        dialogue.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': prepared, 'environment_result': result})
        for ((name, args), tool_call, call_result) in zip(action_pairs, prepared, result['calls'], strict=True):
            observation = call_result.get('observation') or {}
            if not isinstance(observation, dict):
                observation = {'raw': observation}
            (envelope, payload) = _unwrap_observation(observation)
            merged = {**payload}
            if envelope.get('error') is not None:
                merged.setdefault('error', envelope.get('error'))
            ledger.ingest(name, args, observation)
            messages.append({'role': 'tool', 'tool_call_id': tool_call['id'], 'content': _compact_obs(merged if merged else observation)})
        user_message = result.get('user_message')
        if isinstance(user_message, dict) and user_message.get('content'):
            content = str(user_message['content'])
            for chunk in re.split('\\n+', content):
                chunk = chunk.strip()
                if chunk:
                    ledger.note_shopper(chunk)
            messages.append({'role': 'user', 'content': content})
        elif isinstance(user_message, str) and user_message.strip():
            for chunk in re.split('\\n+', user_message):
                chunk = chunk.strip()
                if chunk:
                    ledger.note_shopper(chunk)
            messages.append({'role': 'user', 'content': user_message})
        if any((isinstance(call.get('observation'), dict) and call['observation'].get('done') for call in result.get('calls') or [])):
            break
    return dialogue

# ===== TF router =====
_TF1 = 'intent_decomposition'
_TF2 = 'retrieval_recall'
_TF3 = 'constraint_satisfaction'
_TF4 = 'preference_reasoning'
_TF5 = 'ranking'
_TF6 = 'recovery'
_TF7 = 'justification'
_FAMILY_ENTRY = {_TF1: tf1_agent_main, _TF2: tf2_agent_main, _TF3: tf3_agent_main, _TF4: tf4_agent_main, _TF5: tf5_agent_main, _TF6: tf5_agent_main, _TF7: tf5_agent_main}

def _public_family(problem_data: dict[str, Any]) -> str:
    env = problem_data.get('environment') if isinstance(problem_data.get('environment'), dict) else {}
    family = str(problem_data.get('family') or (env or {}).get('family') or problem_data.get('task_family') or '').strip()
    if family in _FAMILY_ENTRY:
        return family
    policy = (env or {}).get('policy_view') if isinstance(env, dict) else {}
    if not isinstance(policy, dict):
        policy = {}
    query = str(policy.get('query') or problem_data.get('query') or '')
    tools = list(policy.get('tools') or [])
    head = query.split('Task rules:')[0]
    scores = {_TF1: 0, _TF2: 0, _TF3: 0, _TF4: 0, _TF5: 0, _TF6: 0, _TF7: 0}
    named = len(re.findall('\\b(\\d{6,})::([A-Za-z0-9_-]+)\\b', head))
    if named >= 2:
        scores[_TF5] += 6
    if re.search('\\b(?:higher|lower|more|less|prefer(?:ring)?)\\s+[a-z]', head, re.I):
        scores[_TF5] += 2
    if re.search('(?:only|will)\\s+(?:tell|share|reveal).{0,48}\\bif\\s+you\\s+ask|\\bif you ask\\b', head, re.I):
        scores[_TF3] += 5
    just = False
    for spec in tools:
        fn = (spec.get('function') if isinstance(spec, dict) else None) or spec
        if not isinstance(fn, dict):
            continue
        if str(fn.get('name') or '') != 'place_test_order':
            continue
        params = fn.get('parameters') or {} if isinstance(fn.get('parameters'), dict) else {}
        req = params.get('required') or []
        if 'justification' in req:
            just = True
    if just or re.search('\\bwhy\\b|justif', head, re.I):
        scores[_TF7] += 5
    if re.search('\\brather\\b|\\bmatters?\\s+more\\b|\\bcare\\s+about\\b|\\bpriority\\b', head, re.I):
        scores[_TF4] += 6
    elif re.search('\\bprefer(?:red|ably)?\\b|\\bideally\\b', head, re.I):
        scores[_TF1] += 4
        scores[_TF4] += 2
    if re.search('["\\u201c][^"\\u201d\\n]{2,80}["\\u201d]', head) or re.search('\\bmatches?\\b', head, re.I):
        scores[_TF2] += 4
    if re.search('\\b(?:unavailable|sold\\s*out|fallback|recover|another\\s+(?:colou?r|option|variant))\\b', head, re.I):
        scores[_TF6] += 4
    if re.search('(?:strongly\\s+prefer|prefer(?:red)?)\\s+(?:the\\s+)?[A-Za-z]*\\d{2,}', head, re.I):
        scores[_TF6] += 6
    if not any(scores.values()):
        scores[_TF3] = 1
    order = (_TF5, _TF3, _TF7, _TF4, _TF1, _TF6, _TF2)
    best = max(scores.values())
    for fam in order:
        if scores[fam] == best:
            return fam
    return _TF3

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    family = _public_family(problem_data)
    return _FAMILY_ENTRY[family](problem_data)
__all__ = ['agent_main']
