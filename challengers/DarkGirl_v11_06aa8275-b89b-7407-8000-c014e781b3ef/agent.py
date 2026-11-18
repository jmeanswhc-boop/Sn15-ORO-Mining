from __future__ import annotations
import json
import re
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_OPENROUTER_DEFAULT = 'deepseek/deepseek-v3.2'
_OPENROUTER_HARD = 'qwen/qwen3.5-397b-a17b'
_OPENROUTER_FALLBACKS = ('google/gemma-4-31b-it', 'openai/gpt-oss-120b', 'deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-v3.2')
_proxy = ProxyClient(timeout=120, max_retries=2)
_FAMILY_FINGERPRINTS: list[tuple[str, list[re.Pattern[str]]]] = [('ranking', [re.compile('\\btrade-?off\\b.{0,40}\\bpriorit|\\brank(?:ing|ed)?\\b|\\bre-?rank\\b', re.I), re.compile('\\bdominat|\\bstrictly worse\\b|\\bclosed set\\b|\\bnamed candidates?\\b', re.I), re.compile('\\d{8,}::\\S+')]), ('recovery', [re.compile('\\bleast[- ]relax|\\brecover\\b.{0,24}\\bunavailable|\\bfallback\\b', re.I), re.compile('\\bsame model\\b.{0,48}(?:\\bother\\b|\\banother\\b).{0,12}\\bcolou?r\\b', re.I), re.compile('\\bdisappear|\\bsold\\s*out\\b|\\bunavailable\\b', re.I)]), ('justification', [re.compile('\\bjustification\\b.{0,24}\\bclaims?|\\bclaims?\\b.{0,24}\\bjustif|\\bground(?:ed)? claims?\\b', re.I), re.compile("\\btell me why\\b|\\bgive me a reason\\b|\\bwhy\\b.{0,16}(?:the one|it'?s|your pick)|\\bexplain\\s+(?:the\\s+)?(?:choice|pick|order)\\b", re.I)]), ('intent_decomposition', [re.compile('nice[- ]to[- ]have', re.I), re.compile('\\bsubordinate\\b.{0,40}\\b(?:firm|requirement)|preferred brand', re.I), re.compile('\\bfirm\\b.{0,80}\\bsoft\\b|\\bsoft\\b.{0,80}\\bfirm\\b', re.I)]), ('constraint_satisfaction', [re.compile('exact(?:ly)?\\s+(?:varian|sku|option)|specific sku', re.I), re.compile('every (?:firm )?requirement must|\\bgated (?:spec|requirement)\\b', re.I), re.compile('\\bif you ask\\b|\\bwhen asked\\b|\\bwithhold\\b.{0,36}\\bspec\\b', re.I)]), ('preference_reasoning', [re.compile('\\bnot (?:just )?the cheapest\\b|\\bsoft (?:axes|preferences?)\\b', re.I), re.compile('intended use|use[- ]case', re.I), re.compile('\\brather pay\\b|\\bmatters more\\b', re.I)]), ('retrieval_recall', [re.compile('\\breformulate\\b|\\bshortlist\\b', re.I), re.compile('\\bdistinct (?:products|identities)\\b|\\boff[- ]category\\b', re.I), re.compile('matches? (?:the )?(?:phrase )?\\"[^\\"]+\\"', re.I)])]
_ASK_SPEC_RX = re.compile('(?:tell|reveal|share).{0,48}\\bif you ask\\b|\\bif you ask\\b|\\bwhen asked\\b|\\bask\\b.{0,24}(?:exact )?(?:requirement|spec)\\b|\\bwithhold\\b.{0,36}\\bspec\\b', re.I)
_PROMPT_CORE = '\nYou are DarkGirl v7, an ORO Bench shopping agent.\n\n# Contract\n- Use ONLY the supplied tools from the environment.\n- Continue until an observation reports done=true.\n- Respect max_steps and max_calls_per_turn.\n- Prefer parallel tool calls when the limit allows and it saves steps.\n\n# Healthy loop\n1) Discover — search / filter\n2) Inspect — view / compare / inspect_stock\n3) Decide — add_to_cart / remove_from_cart\n4) Confirm — inspect_stock / inspect_cart on the final SKU\n5) Finish — place_test_order (with justification.claims when required)\n\n# Precision discipline (from champion select prompts)\nA candidate is valid ONLY if every FIRM constraint from the shopper goal is\nsatisfied on the EXACT variant you order.\n\n## Constraint extraction\nExtract constraints into categories such as:\nbrand, model, series, identifier/SKU, size, capacity, color, material,\nfeature, compatibility, category, budget/price, stock, any named\nspec/attribute gate in the goal, use-case preference.\n\nMark each as FIRM (must hold) or SOFT (preference / nice-to-have) using the\nTask rules. Soft preferences never override firm ones.\n\n## Matching modes\n- Literal: brands, model codes, identifiers, uncommon tokens, exact sizes,\n  colors named as codes. Must appear explicitly (casing may differ).\n- Size/unit wording: keep the shopper\'s own tokens; do not invent conversions.\n- Semantic: everyday paraphrases when Task rules allow them.\n  Prefer false positives over false negatives for soft semantic checks.\n- Spec classes: If Task rules define class floors, follow those; otherwise\n  compare numeric evidence only — never invent thresholds.\n\n## Search / keyword refine (from champion review prompts)\nCatalog search matches VALUES, not JSON keys. Prefer value tokens:\nproduct type words, brand, model codes, distinctive phrase fragments.\nAvoid dumping field names like "brand" or "category" as keywords.\nHyphens ≈ spaces. Morphological variants usually match.\nIf the first search is weak, make ONE small reformulation (drop noise,\nkeep distinctive tokens) — do not rewrite everything.\n\n# Messaging\nUse message when Task rules require clarifying priorities or revealing a\nhidden firm spec. Keep messages short.\n\n# Shopper interventions\nIf user_message is a question, hold, or "which one":\ncall the message tool THIS turn with a concrete answer, then continue.\nIgnoring a question fails even if the final SKU is valid.\n\n# Market / cart events\nAfter stock_changed, price_changed, a failed cart, or remove_from_cart:\nre-inspect price/stock, recover per family rules, then place_test_order\nso the event is closed. Unfinished events score poorly.\n\n# First cart is graded\nDo not cart a placeholder. Leave ≥1–2 turns after a shock for inspect + order.\nLate turns: stop open search and finish.\n'.strip()
_PROMPT_FAMILY: dict[str, str] = {'intent_decomposition': '\n# Family: TF1 Intent Decomposition\nRefs: Ostrich / IamSUPER (decompose firm vs soft, then act).\n\nPlan a short decision sequence before committing:\n1. List firm requirements (category, named spec/attribute, budget, in-stock).\n2. List soft preferences (named preferred brand).\n3. Dual shortlist: (brand+firm) and (firm-only backup).\n4. First cart: preferred brand IF a fully compliant SKU exists;\n   otherwise the best firm-compliant item. No placeholders.\n5. If the shopper asks which/spec/budget, message with evidence, then continue.\n6. After price/stock shocks: remove invalid, recover under the new world with\n   the same brand rule, inspect, then place_test_order to close the event.\n\nThe first cart add is scored separately from later fixes.\n'.strip(), 'retrieval_recall': '\n# Family: TF2 Retrieval Recall\nRefs: Paris / Bolt-2026 (precision select + keyword refine).\n\n- Reformulate queries; try phrase, tokens, and category+filter.\n- Survey several DISTINCT products before buying — avoid committing after\n  a single weak hit.\n- View/compare promising hits; reject accessories and off-category noise.\n- Once a solid shortlist exists, buy the best in-stock item within budget.\n- Do not wait for permission after a verified in-stock, in-budget pick.\n'.strip(), 'constraint_satisfaction': '\n# Family: TF3 Constraint Satisfaction\nRefs: Storm / Slackbro (exact-variant discipline).\n\n- EVERY firm requirement must hold on the EXACT SKU ordered.\n- Titles often mix compliant and non-compliant variants — always view, then\n  inspect_stock on the chosen SKU (price + stock + option).\n- If the shopper withholds a firm spec until asked, call message FIRST to\n  request that requirement before searching. Do not search until they reply.\n- Apply any class/threshold rules written in the Task rules; never invent floors.\n- After sellout: a NEW product_id that still meets every firm gate — not a\n  sibling option that drops a constraint. Always finish with place_test_order.\n'.strip(), 'preference_reasoning': '\n# Family: TF4 Preference Reasoning\nRefs: sangaria / IamSUPER (tradeoffs, not cheapest-by-default).\n\nHard: category, budget, in-stock, real device (not accessory/review page).\nSoft: use-case preferences the shopper actually named. Do not invent extra axes.\n\nBefore the first cart add:\n- Optionally message to clarify priorities.\n- compare/view candidates across soft axes.\n- Prefer a slightly dearer item that matches soft goals if still in budget.\n'.strip(), 'ranking': '\n# Family: TF5 Ranking\nRefs: PriceMeridian / boston (ordered tradeoffs).\n\n- Restrict to the candidate variants named in the query (product_id::sku).\n  Prefer viewing those IDs directly; do not burn the step budget on open search.\n- Rank ONLY by the tradeoff priority stated in the Task rules / goal.\n  Never invent extra axes or default axis orders.\n- View/compare facts before the first cart add; the first cart choice must be\n  the best-ranked AVAILABLE option — never a placeholder.\n- After stockout / price / cart events (often after remove_from_cart or a\n  failed add): re-inspect remaining candidates, RE-RANK from scratch with the\n  same priority, cart the new #1, inspect_stock, then place_test_order.\n  Do NOT keep or re-order the pre-event #1 if it is no longer best/available.\n- Never pick an option that is strictly dominated on every stated axis.\n- Always finish with place_test_order before max_steps.\n\nCRITICAL — place_test_order:\n- Pass ONLY product_id and sku.\n- Do NOT include justification/claims unless the Task rules require them.\n'.strip(), 'recovery': '\n# Family: TF6 Recovery\nRefs: Abudullah (least-relaxation recovery).\n\nNever relax: category, budget, in-stock.\nLeast-relaxation ladder (do not skip levels):\n0) exact model + exact color\n1) same model, other color\n2) other models last\nIf the preferred item disappears after a cart/stock change, climb one level,\nre-check price/stock, then place_test_order to close the event.\nUse explicit model facts when supplied; conflicting titles do not override them.\nOmit optional justification/claims unless the goal asks for a reason.\n'.strip(), 'justification': '\n# Family: TF7 Justification\n\n- View/compare raw facts before the first cart add and before ordering.\n- Do not justify from marketing title words alone.\n- place_test_order MUST include justification.claims as field/value pairs.\n- Claim `field` must be a JSON key from view/compare/inspect_stock — copy\n  those keys. Never use marketing/meta labels (display, title, category, OS).\n  When those facts exist, include price, in_stock, the spec key the shopper\n  asked about (the same key as in the observation), and at most one extra\n  evidenced reason (prefer brand).\n- Do not invent feature flags. Do not remap via alias tables.\n- Claim `value` must equal viewed / inspect_stock facts. Do not rewrite\n  measurements into a different unit or layout.\n- Ground every claim in viewed or inspect_stock/cart state.\n- Keep claims minimal. No summary/evidence prose inside claims.\n'.strip()}
_STUCK_NUDGES = ('No tool call received. Choose a supplied tool now. If you have not searched yet, call search or filter.', 'Still no tool call. Progress the episode: search/filter, view a candidate, or place_test_order if the cart already holds a verified in-stock SKU.')
_COACH_ASK_SPEC = 'Coach: a firm requirement appears to be withheld until you ask. Call message first to request that exact requirement before searching. Do not search or filter until the shopper replies.'
_COACH_RANK_RERANK = 'Coach [TF5]: market/cart state changed. Re-inspect the named candidate variants that are still available, re-rank by the stated tradeoff priority, cart the NEW #1 (not the pre-event pick), inspect_stock, then place_test_order with only product_id+sku.'
_COACH_RANK_FINISH = 'Coach [TF5]: stop searching. Re-rank the named candidates still in stock, add the current #1 to cart if needed, and place_test_order now (product_id+sku only).'
_COACH_TF7_CLAIMS = 'Coach [TF7]: on place_test_order, justification.claims must copy catalog JSON keys from view/inspect_stock — price, in_stock, the asked spec key, and at most one evidenced extra (prefer brand). Drop marketing/meta labels (display, title, category, OS, summary, evidence). Values must equal observed catalog text; do not invent features or rewrite units.'
_COACH_REPLY = 'Coach: the shopper just spoke. Call message THIS turn with a concrete answer (identity + key firm evidence + price/stock). Do not cart or order until that reply is sent.'
_COACH_BRAND = 'Coach [TF1]: keep a brand+firm shortlist and a firm-only backup. If a preferred-brand SKU is fully firm-compliant, it MUST win the first cart; otherwise cart the best firm-compliant any-brand item. Answer shopper questions with message before further cart/order.'
_COACH_MARKET = 'Coach: catalog/cart state moved ({kind}). Re-check price/stock on the affected SKU. If invalid, remove and recover per the family rules, then place_test_order so the event is closed.'
_COACH_FINISH = 'Coach: stop open search. Verify the carted SKU (or cart the current valid pick), then place_test_order now so the episode closes.'
_COACH_RECOVERY = 'Coach: an action failed or stock/price changed ({error}). Climb the least-relaxation ladder without skipping: exact model+color → same model other color → other models last. Never relax category, budget, or in-stock. Then place_test_order to close the event.'
_COACH_TF3_SELLOUT = 'Coach [TF3]: the current SKU is gone. Switch to a NEW product_id that still meets every firm constraint — sibling options that drop a firm spec are invalid. Then place_test_order.'
_COACH_TF6_LADDER = 'Coach [TF6]: climb the least-relaxation ladder without skipping — same model / other color before other models. Never relax category, budget, or in-stock. Omit optional claims. Then place_test_order.'
_CLAIM_META_DENY = frozenset({'category', 'category_path', 'title', 'summary', 'evidence', 'reason', 'description', 'name', 'note', 'display', 'device_operating_system', 'device os', 'operating_system', 'operating system', 'os'})
_VARIANT_ID_RX = re.compile('\\b(\\d{10,}::[A-Za-z0-9._-]+)\\b')

def _model_for(family: str) -> str:
    override = (getenv('SANDBOX_MODEL') or '').strip()
    if override:
        mapped = _map_chutes_to_openrouter(override)
        return mapped
    hard = {'constraint_satisfaction', 'preference_reasoning', 'ranking', 'recovery', 'justification', 'intent_decomposition'}
    return _OPENROUTER_HARD if family in hard else _OPENROUTER_DEFAULT

def _map_chutes_to_openrouter(model: str) -> str:
    table = {'deepseek-ai/DeepSeek-V3.2-TEE': 'deepseek/deepseek-v3.2', 'deepseek-ai/DeepSeek-V3.1-TEE': 'deepseek/deepseek-chat-v3.1', 'deepseek-ai/DeepSeek-V3-0324-TEE': 'deepseek/deepseek-chat-v3-0324', 'deepseek-ai/DeepSeek-R1-0528-TEE': 'deepseek/deepseek-r1-0528', 'Qwen/Qwen3-32B-TEE': 'qwen/qwen3-32b', 'Qwen/Qwen3.5-397B-A17B-TEE': 'qwen/qwen3.5-397b-a17b', 'Qwen/Qwen3.6-27B-TEE': 'qwen/qwen3.6-27b', 'google/gemma-4-31B-turbo-TEE': 'google/gemma-4-31b-it', 'openai/gpt-oss-120b-TEE': 'openai/gpt-oss-120b', 'XiaomiMiMo/MiMo-V2-Flash-TEE': 'xiaomi/mimo-v2-flash', 'unsloth/Mistral-Nemo-Instruct-2407-TEE': 'mistralai/mistral-small-2603', 'zai-org/GLM-5.1-TEE': 'qwen/qwen3.5-397b-a17b', 'zai-org/GLM-5.2-TEE': 'qwen/qwen3.5-397b-a17b', 'moonshotai/Kimi-K2.6-TEE': 'qwen/qwen3.5-397b-a17b', 'moonshotai/Kimi-K2.5-TEE': 'qwen/qwen3.5-397b-a17b', 'moonshotai/kimi-k2.6': 'qwen/qwen3.5-397b-a17b', 'z-ai/glm-5.1': 'qwen/qwen3.5-397b-a17b', 'z-ai/glm-5.2': 'qwen/qwen3.5-397b-a17b'}
    mapped = table.get(model, model)
    low = mapped.lower()
    if 'kimi' in low or 'glm' in low or low.startswith('z-ai/') or low.startswith('moonshotai/'):
        return _OPENROUTER_HARD
    return mapped

def _detect_family(query: str) -> str:
    text = query or ''
    scores: dict[str, int] = {}
    for (fam, patterns) in _FAMILY_FINGERPRINTS:
        hits = sum((1 for rx in patterns if rx.search(text)))
        if hits:
            scores[fam] = hits
    if 'justification is optional' in text.lower() and scores.get('justification'):
        goal = text.split('Task rules:')[0] if 'Task rules:' in text else text
        if not re.search('\\bwhy\\b|reason|justify', goal, re.I):
            scores.pop('justification', None)
    if not scores:
        return 'retrieval_recall'
    priority = {'ranking': 7, 'recovery': 6, 'justification': 5, 'constraint_satisfaction': 4, 'intent_decomposition': 3, 'preference_reasoning': 2, 'retrieval_recall': 1}
    return max(scores.items(), key=lambda kv: (kv[1], priority.get(kv[0], 0)))[0]

def _shopper_goal(query: str) -> str:
    text = query or ''
    return text.split('Task rules:')[0] if 'Task rules:' in text else text

def _needs_ask_spec(query: str) -> bool:
    return bool(_ASK_SPEC_RX.search(_shopper_goal(query)))

def _goal_asks_why(query: str) -> bool:
    return bool(re.search('\\bwhy\\b|justif|reason|tell me why', _shopper_goal(query), re.I))

def _named_variants(query: str) -> list[str]:
    return list(dict.fromkeys(_VARIANT_ID_RX.findall(query or '')))

def _shopper_needs_reply(text: str) -> bool:
    blob = (text or '').strip().lower()
    if not blob:
        return False
    if '?' in blob:
        return True
    cues = ('^\\s*which\\b', '^\\s*what\\b', '^\\s*does\\b', '^\\s*did\\b', '\\bcan you\\b', '\\bcould you\\b', '^\\s*wait\\b', '\\bhold on\\b', '\\bbefore you\\b', '\\bre-?check\\b', '\\bgot anything\\b')
    return any((re.search(pat, blob) for pat in cues))

def _market_kind(obs: Any) -> str:
    if not isinstance(obs, dict):
        return ''
    if obs.get('stock_changed'):
        return 'stock_changed'
    if obs.get('price_changed'):
        return 'price_changed'
    if obs.get('event'):
        return str(obs.get('event'))
    node = obs.get('market_event')
    if isinstance(node, dict):
        return str(node.get('kind') or 'market_event')
    if node:
        return 'market_event'
    nested = obs.get('observation')
    if isinstance(nested, dict):
        return _market_kind(nested)
    return ''

def _observed_catalog_keys(obs_list: list[Any], *, limit: int=12) -> list[str]:
    skip = _CLAIM_META_DENY | {'error', 'done', 'calls', 'observation', 'event', 'stock_changed', 'price_changed', 'products', 'results', 'items', 'product_id', 'sku', 'options', 'sku_options', 'market_event'}
    keys: list[str] = []
    seen: set[str] = set()

    def walk(node: Any, depth: int=0) -> None:
        if depth > 6 or len(keys) >= limit:
            return
        if isinstance(node, dict):
            for (raw_key, value) in node.items():
                key = str(raw_key).strip()
                low = key.lower()
                if not key or low in skip:
                    walk(value, depth + 1)
                    continue
                if low not in seen and isinstance(value, (str, int, float, bool)):
                    seen.add(low)
                    keys.append(key)
                else:
                    walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node[:12]:
                walk(item, depth + 1)
    for obs in obs_list:
        walk(obs)
    return keys[:limit]

def _tf7_claims_coach(last_obs: list[Any]) -> str:
    keys = _observed_catalog_keys(obs_list=last_obs)
    extra = ''
    if keys:
        extra = ' Observed catalog keys from the last view/inspect: ' + ', '.join(keys) + '. Copy those keys as claim fields.'
    return _COACH_TF7_CLAIMS + extra

def _normalize_claim_field(field: str, query: str) -> str:
    del query
    raw = str(field or '').strip()
    if raw.lower() in _CLAIM_META_DENY:
        return ''
    return raw

def _normalize_claim_value(field: str, value: Any) -> str:
    text = str(value).strip() if value is not None else ''
    if field == 'in_stock':
        return 'true' if text.lower() in {'true', '1', 'yes', 'in_stock'} else text
    return text

def _sanitize_tool_args(name: str, args: dict[str, Any], *, family: str, query: str) -> dict[str, Any]:
    out = dict(args)
    if name != 'place_test_order':
        return out
    if family != 'justification':
        if not _goal_asks_why(query):
            out.pop('justification', None)
        return out
    just = out.get('justification')
    if not isinstance(just, dict):
        return out
    claims = just.get('claims')
    if not isinstance(claims, list):
        return out
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get('field') or ''), query)
        value = _normalize_claim_value(field, claim.get('value'))
        if not field or value == '':
            continue
        fl = field.lower()
        if fl in _CLAIM_META_DENY:
            continue
        if fl in seen:
            continue
        seen.add(fl)
        cleaned.append({'field': field, 'value': value})
    if cleaned:
        out['justification'] = {'claims': cleaned}
    return out

def _tool_name_list(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        fn = tool.get('function') if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get('name'):
            names.append(str(fn['name']))
        elif isinstance(tool, dict) and tool.get('name'):
            names.append(str(tool['name']))
    return names

def _system_prompt(family: str, policy: dict[str, Any], query: str) -> str:
    names = _tool_name_list(policy.get('tools') or [])
    family_prompt = _PROMPT_FAMILY.get(family, _PROMPT_FAMILY['retrieval_recall'])
    variants = _named_variants(query)
    variant_line = ''
    if family == 'ranking' and variants:
        variant_line = 'Named candidate variants (view these; re-rank after events):\n' + ', '.join(variants) + '\n'
    return f"{_PROMPT_CORE}\n\n{family_prompt}\n\n{variant_line}# Runtime\nDetected task family: {family}\nAvailable tools: {(', '.join(names) if names else '(see tool schemas)')}\nmax_steps={policy.get('max_steps')} max_calls_per_turn={policy.get('max_calls_per_turn', 1)}\n"

def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call['function'].get('arguments', '{}')
    if isinstance(raw, dict):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw or '{}')
        except json.JSONDecodeError:
            parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError('tool arguments must be a JSON object')
    return parsed

def _compact_observation(obs: Any, limit: int=6000) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    if len(text) <= limit:
        return text
    return text[:limit - 20] + '...<truncated>'

def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    names = []
    for call in tool_calls:
        try:
            names.append(str(call['function']['name']))
        except (KeyError, TypeError):
            names.append('?')
    return names

def _progress_hint(family: str, turn: int, last_tool_names: list[str], last_obs: list[Any], query: str, *, max_steps: int=20, await_reply: bool=False, saw_cart_event: bool=False, ordered: bool=False, first_cart_done: bool=False, asked_hidden_spec: bool=False) -> str | None:
    joined = ' '.join(last_tool_names)
    if await_reply:
        return _COACH_REPLY
    if family == 'constraint_satisfaction' and _needs_ask_spec(query) and (not asked_hidden_spec) and ('message' not in joined):
        return _COACH_ASK_SPEC
    if family == 'ranking' and (not ordered):
        if saw_cart_event and turn >= 2:
            if 'place_test_order' not in joined:
                return _COACH_RANK_RERANK
        if turn >= 8:
            return _COACH_RANK_FINISH
        if turn == 1:
            return 'Hint [TF5]: view/compare the named candidate variants and rank by the stated tradeoff axes before add_to_cart. After any stock/cart event, re-rank and order the new #1. place_test_order = product_id+sku only.'
        if turn >= 2 and (not first_cart_done) and ('view' in joined or 'compare' in joined):
            return 'Coach [TF5]: candidates inspected — add_to_cart only the current #1 (first cart is graded).'
    if family == 'intent_decomposition' and turn == 1:
        return _COACH_BRAND
    if family == 'justification' and (not ordered):
        if turn in (2, 5) or (turn >= 2 and 'view' not in joined and ('compare' not in joined)):
            return _tf7_claims_coach(last_obs)
    if family == 'recovery' and saw_cart_event and (not ordered):
        return _COACH_TF6_LADDER
    if family == 'constraint_satisfaction' and saw_cart_event and (not ordered):
        return _COACH_TF3_SELLOUT
    if not ordered and turn >= max(8, max_steps - 2):
        return _COACH_FINISH
    if turn <= 2 and (not any((n in joined for n in ('search', 'filter', 'view', 'compare', 'message')))):
        return 'Hint: start with search or filter using value-side keywords from the shopper goal, then view the best hits.'
    if family == 'recovery' and any((isinstance(o, dict) and o.get('error') for o in last_obs)):
        return _COACH_RECOVERY.format(error=str(next((o.get('error') for o in last_obs if isinstance(o, dict) and o.get('error')))))
    if family == 'retrieval_recall' and turn == 3 and ('search' in joined):
        return 'Hint: make one small keyword reformulation or add a filter so the shortlist covers several distinct products before ordering.'
    if family == 'preference_reasoning' and turn == 2 and ('compare' not in joined):
        return 'Hint: compare candidates on soft preference axes (not only price) before the first cart add.'
    if saw_cart_event and (not ordered) and (turn >= 2):
        return _COACH_MARKET.format(kind='stock/price/cart')
    return None

def _infer(*, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_calls: int) -> tuple[str, list[dict[str, Any]]]:
    models_to_try = [model]
    for fb in _OPENROUTER_FALLBACKS:
        if fb not in models_to_try:
            models_to_try.append(fb)
    last_err: Exception | None = None
    for mid in models_to_try[:3]:
        inference = _proxy.post('/inference/chat/completions', json_data={'model': mid, 'messages': messages, 'tools': tools, 'tool_choice': 'required', 'temperature': 0})
        if inference is None:
            last_err = RuntimeError(f'inference failed for model={mid}')
            continue
        try:
            assistant = inference['choices'][0]['message']
        except (KeyError, IndexError, TypeError) as exc:
            last_err = exc
            continue
        content = assistant.get('content') or ''
        tool_calls = list(assistant.get('tool_calls') or [])[:max_calls]
        return (content, tool_calls)
    raise RuntimeError(f'inference request failed ({last_err})')

def _opening_coaches(family: str, query: str) -> list[dict[str, Any]]:
    extras: list[dict[str, Any]] = []
    if family == 'constraint_satisfaction' and _needs_ask_spec(query):
        extras.append({'role': 'user', 'content': _COACH_ASK_SPEC})
    if family == 'ranking':
        extras.append({'role': 'user', 'content': 'Hint [TF5]: view/compare the named candidate variants and rank by the stated tradeoff axes before add_to_cart. After any stock/cart event, re-rank and order the new #1. place_test_order = product_id+sku only.'})
    if family == 'justification':
        extras.append({'role': 'user', 'content': _COACH_TF7_CLAIMS})
    if family == 'intent_decomposition':
        extras.append({'role': 'user', 'content': _COACH_BRAND})
    return extras

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data['environment']
    binding = environment['binding']
    policy = environment['policy_view']
    query = str(policy.get('query') or problem_data.get('query') or '')
    problem_id = str(problem_data.get('problem_id', problem_data.get('id', 'problem')))
    max_calls = max(1, int(policy.get('max_calls_per_turn', 1)))
    max_steps = max(1, int(policy.get('max_steps', 20)))
    tools = policy.get('tools') or []
    family = _detect_family(query)
    model = _model_for(family)
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': _system_prompt(family, policy, query)}, {'role': 'user', 'content': query}, *_opening_coaches(family, query)]
    dialogue: list[dict[str, Any]] = [{'role': 'meta', 'darkgirl': {'version': 7, 'family': family, 'model': model, 'provider': 'openrouter'}}]
    last_tool_names: list[str] = []
    last_obs: list[Any] = []
    await_reply = False
    saw_cart_event = False
    ordered = False
    first_cart_done = False
    asked_hidden_spec = False
    for turn in range(1, max_steps + 1):
        hint = _progress_hint(family, turn, last_tool_names, last_obs, query, max_steps=max_steps, await_reply=await_reply, saw_cart_event=saw_cart_event, ordered=ordered, first_cart_done=first_cart_done, asked_hidden_spec=asked_hidden_spec)
        if hint:
            messages.append({'role': 'user', 'content': hint})
        assistant_content = ''
        tool_calls: list[dict[str, Any]] = []
        for attempt in range(len(_STUCK_NUDGES) + 1):
            (assistant_content, tool_calls) = _infer(model=model, messages=messages, tools=tools, max_calls=max_calls)
            if tool_calls:
                break
            dialogue.append({'role': 'assistant', 'content': assistant_content})
            nudge = _STUCK_NUDGES[min(attempt, len(_STUCK_NUDGES) - 1)]
            messages.extend([{'role': 'assistant', 'content': assistant_content}, {'role': 'user', 'content': nudge}])
        else:
            raise RuntimeError('model returned no tool call before completion')
        messages.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls})
        group_id = f'{problem_id}-turn-{turn}'
        call_payloads = []
        for (index, call) in enumerate(tool_calls, start=1):
            name = call['function']['name']
            args = _sanitize_tool_args(name, _arguments(call), family=family, query=query)
            if isinstance(call.get('function'), dict):
                call['function']['arguments'] = json.dumps(args)
            call_payloads.append({'call_id': f'{group_id}-{index}', 'action': {'name': name, 'args': args}})
            if name == 'message':
                await_reply = False
                asked_hidden_spec = True
            elif name == 'add_to_cart':
                first_cart_done = True
            elif name == 'remove_from_cart':
                saw_cart_event = True
            elif name == 'place_test_order':
                ordered = True
        envelope = {**binding, 'call_id': group_id, 'idempotency_key': group_id, 'turn': turn, 'calls': call_payloads}
        result = _proxy.post('/environment/call', json_data=envelope)
        if result is None:
            raise RuntimeError('environment call failed')
        dialogue.append({'role': 'assistant', 'content': assistant_content, 'tool_calls': tool_calls, 'environment_result': result, 'darkgirl_turn': {'family': family, 'turn': turn, 'await_reply': await_reply, 'saw_cart_event': saw_cart_event, 'ordered': ordered}})
        last_tool_names = _tool_names(tool_calls)
        last_obs = []
        for (tool_call, call_result) in zip(tool_calls, result['calls'], strict=True):
            obs = call_result.get('observation')
            last_obs.append(obs)
            messages.append({'role': 'tool', 'tool_call_id': tool_call['id'], 'content': _compact_observation(obs)})
            kind = _market_kind(obs)
            if kind:
                saw_cart_event = True
                if not ordered:
                    messages.append({'role': 'user', 'content': _COACH_MARKET.format(kind=kind)})
            elif isinstance(obs, dict) and obs.get('error'):
                saw_cart_event = True
        user_message = result.get('user_message')
        if user_message and user_message.get('content'):
            spoken = str(user_message['content'])
            messages.append({'role': 'user', 'content': spoken})
            if _shopper_needs_reply(spoken):
                await_reply = True
        errors = [o.get('error') for o in last_obs if isinstance(o, dict) and o.get('error')]
        if errors and family == 'recovery' and (not ordered):
            messages.append({'role': 'user', 'content': _COACH_RECOVERY.format(error=errors[0])})
        if family == 'ranking' and saw_cart_event and (not ordered) and (errors or any((_market_kind(o) for o in last_obs))):
            messages.append({'role': 'user', 'content': _COACH_RANK_RERANK})
        if any((isinstance(call.get('observation'), dict) and call['observation'].get('done') for call in result.get('calls') or [])):
            break
    return dialogue
__all__ = ['agent_main']
