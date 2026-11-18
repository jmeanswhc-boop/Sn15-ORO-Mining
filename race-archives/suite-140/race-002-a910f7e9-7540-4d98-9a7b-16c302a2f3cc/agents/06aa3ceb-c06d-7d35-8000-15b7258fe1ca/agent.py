from __future__ import annotations
import json
import re
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_MODEL_LIGHT = 'deepseek/deepseek-v3.2'
_MODEL_DENSE = 'qwen/qwen3.5-397b-a17b'
_MODEL_FALLBACK = ('google/gemma-4-31b-it', 'openai/gpt-oss-120b', 'deepseek/deepseek-chat-v3.1')
_CHUTES_TO_OR = {'deepseek-ai/DeepSeek-V3.2-TEE': 'deepseek/deepseek-v3.2', 'deepseek-ai/DeepSeek-V3.1-TEE': 'deepseek/deepseek-chat-v3.1', 'deepseek-ai/DeepSeek-V3-0324-TEE': 'deepseek/deepseek-chat-v3-0324', 'deepseek-ai/DeepSeek-R1-0528-TEE': 'deepseek/deepseek-r1-0528', 'Qwen/Qwen3-32B-TEE': 'qwen/qwen3-32b', 'Qwen/Qwen3.5-397B-A17B-TEE': 'qwen/qwen3.5-397b-a17b', 'Qwen/Qwen3.6-27B-TEE': 'qwen/qwen3.6-27b', 'google/gemma-4-31B-turbo-TEE': 'google/gemma-4-31b-it', 'openai/gpt-oss-120b-TEE': 'openai/gpt-oss-120b', 'XiaomiMiMo/MiMo-V2-Flash-TEE': 'xiaomi/mimo-v2-flash', 'unsloth/Mistral-Nemo-Instruct-2407-TEE': 'mistralai/mistral-small-2603', 'zai-org/GLM-5.1-TEE': 'qwen/qwen3.5-397b-a17b', 'zai-org/GLM-5.2-TEE': 'qwen/qwen3.5-397b-a17b', 'moonshotai/Kimi-K2.6-TEE': 'qwen/qwen3.5-397b-a17b', 'moonshotai/Kimi-K2.5-TEE': 'qwen/qwen3.5-397b-a17b'}
_proxy = ProxyClient(timeout=120, max_retries=2)
_OBS_CEILING = 6000
STRAND_BRAID = 'braid'
STRAND_SCOUT = 'scout'
STRAND_CLAMP = 'clamp'
STRAND_SAVOR = 'savor'
STRAND_LADDER = 'ladder'
STRAND_RECAST = 'recast'
STRAND_ATTEST = 'attest'
_DENSE_STRANDS = frozenset({STRAND_BRAID, STRAND_CLAMP, STRAND_SAVOR, STRAND_LADDER, STRAND_RECAST, STRAND_ATTEST})
BEAT_PLUCK = 'pluck'
BEAT_TENSION = 'tension'
BEAT_KNOT = 'knot'
BEAT_TRIM = 'trim'
BIT_CLOSED = 1 << 0
BIT_ASK = 1 << 1
BIT_WHY = 1 << 2
BIT_RECOVER = 1 << 3
BIT_FIRM_SOFT = 1 << 4
BIT_PREFER = 1 << 5
BIT_QUOTE = 1 << 6
BIT_EXACT = 1 << 7
BIT_MULTI = 1 << 8
_LATCH_PATTERNS: tuple[tuple[int, str], ...] = ((BIT_CLOSED, STRAND_LADDER), (BIT_RECOVER, STRAND_RECAST), (BIT_WHY, STRAND_ATTEST), (BIT_ASK | BIT_EXACT, STRAND_CLAMP), (BIT_ASK, STRAND_CLAMP), (BIT_EXACT, STRAND_CLAMP), (BIT_FIRM_SOFT | BIT_MULTI, STRAND_BRAID), (BIT_FIRM_SOFT, STRAND_BRAID), (BIT_PREFER, STRAND_SAVOR), (BIT_QUOTE, STRAND_SCOUT), (BIT_MULTI, STRAND_BRAID), (0, STRAND_SCOUT))
_STRAND_OWNS: dict[str, int] = {STRAND_LADDER: BIT_CLOSED, STRAND_RECAST: BIT_RECOVER, STRAND_ATTEST: BIT_WHY, STRAND_CLAMP: BIT_ASK | BIT_EXACT, STRAND_BRAID: BIT_FIRM_SOFT | BIT_MULTI, STRAND_SAVOR: BIT_PREFER, STRAND_SCOUT: BIT_QUOTE}
_MUTEX: frozenset[frozenset[str]] = frozenset({frozenset({STRAND_LADDER, STRAND_SCOUT})})
_BIT_TO_WARP: tuple[tuple[int, str], ...] = ((BIT_ASK, STRAND_CLAMP), (BIT_WHY, STRAND_ATTEST), (BIT_RECOVER, STRAND_RECAST), (BIT_CLOSED, STRAND_LADDER), (BIT_FIRM_SOFT, STRAND_BRAID), (BIT_PREFER, STRAND_SAVOR), (BIT_QUOTE, STRAND_SCOUT), (BIT_EXACT, STRAND_CLAMP))
_IDLE_NUDGES = ('No executable tool came back. Choose one of the supplied tools and act (search/filter, view, message, or place_test_order).', 'Still idle. Advance the loom: discover, bind an exact sku, or close with place_test_order once the cart is verified.')
_RX_VARIANT = re.compile('(?<!\\d)(\\d{6,})::([A-Za-z0-9._-]+)')
_RX_QUOTE = re.compile('[\\"\\u201c](?P<body>[^\\"\\u201d]{3,55})[\\"\\u201d]')
_RX_CEILING = re.compile('(?:budget|cap|ceiling|at most|no more than|under|below|up to|max(?:imum)?(?:\\s+(?:of|price))?)\\s*[:=]?\\s*([\\d][\\d,]*(?:\\.\\d+)?)', re.I)
_RX_ASK = re.compile('(?:only|will)\\s+(?:tell|share|reveal).{0,48}\\bif\\s+you\\s+ask|\\bif you ask\\b|\\bwhen asked\\b|ask\\s+(?:me\\s+)?(?:for\\s+)?(?:the\\s+)?(?:exact\\s+)?(?:figure|spec|requirement)', re.I)
_RX_WHY = re.compile('\\bwhy\\s+(?:this|that|your|the|it)\\b|\\btell me why\\b|\\bgive me a reason\\b|\\bwalk me through why\\b|\\breason\\s+(?:for|i can|i should)\\b|\\bjustif(?:y|ication)\\b|\\bexplain\\s+(?:the\\s+)?(?:choice|pick|order)\\b|claims?\\s+must\\s+be\\s+grounded|include\\s+justification|\\bstructured justification\\b|\\btrustworthy reason\\b', re.I)
_RX_OPTIONAL_WHY = re.compile('claims?\\s+(?:are|is)\\s+optional|justification\\s+optional', re.I)
_RX_RANK = re.compile('\\btrade-?offs?\\b|\\brank(?:ing|ed)?\\b|\\bre-?rank\\b|\\bdominat(?:e|ed|es)\\b|strictly\\s+worse|\\bcandidate\\s+variants?\\b|\\blexico', re.I)
_RX_RECOVER = re.compile('\\bunavailable\\b|\\bsold\\s*out\\b|\\bdisappear(?:s|ed)?\\b|least[\\s-]?relax(?:ation)?|\\bfallback\\b|other\\s+(?:colour|color)\\b|\\banother\\s+model\\b', re.I)
_RX_FIRM_SOFT = re.compile('\\bfirm\\b.{0,90}\\b(?:soft|wish|prefer)|\\b(?:soft|wish)\\b.{0,90}\\bfirm\\b|nice[\\s-]to[\\s-]have|\\bsubordinate\\b|preferred\\s+brand|initial\\s+cart', re.I)
_RX_PREFER = re.compile('\\brather\\s+pay\\b|\\bintended\\s+use\\b|\\buse[\\s-]case\\b|not\\s+(?:just\\s+)?the\\s+cheapest|preference\\s+axes|matters\\s+more', re.I)
_RX_EXACT = re.compile('exact\\s+(?:variant|sku|option)|specific\\s+sku|every\\s+(?:firm\\s+)?requirement', re.I)
_RX_MULTI = re.compile('\\bfirst[,:].{0,80}\\bsecond\\b|\\bitem\\s*[#:]?\\s*[123]\\b|these\\s+items|all\\s+these|a\\s+shop\\s+(?:that\\s+)?(?:offers|selling)', re.I)
_FIELD_DENY = frozenset({'description', 'note', 'reason', 'evidence', 'summary', 'name', 'title', 'category_path', 'category', 'os'})

def _split_brief(text: str) -> tuple[str, str]:
    (head, sep, tail) = (text or '').partition('\n\n')
    if not sep:
        return ((text or '').strip(), '')
    body = tail.strip()
    (first, _, rest) = body.partition('\n')
    if first.rstrip().endswith(':') and len(first) < 48:
        body = rest.strip()
    return (head.strip(), body)

def _uniq(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))

def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        fn = tool.get('function') if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get('name'):
            names.append(str(fn['name']))
        elif isinstance(tool, dict) and tool.get('name'):
            names.append(str(tool['name']))
    return names

def _popcount(mask: int) -> int:
    return bin(mask).count('1')

def _mutex_blocks(a: str, b: str) -> bool:
    return frozenset({a, b}) in _MUTEX

class LatchBoard:
    __slots__ = ('raw', 'shopper', 'ops', 'variants', 'quote', 'ceiling', 'mask', 'tool_names', 'root', 'warps', 'trace', 'tension_unlocked')

    def __init__(self, query: str, tools: list[dict[str, Any]]) -> None:
        self.raw = query or ''
        (self.shopper, self.ops) = _split_brief(self.raw)
        self.variants = _uniq((f'{m.group(1)}::{m.group(2)}' for m in _RX_VARIANT.finditer(self.raw)))
        qhit = _RX_QUOTE.search(self.shopper)
        self.quote = qhit.group('body') if qhit else ''
        chit = _RX_CEILING.search(self.shopper) or _RX_CEILING.search(self.raw)
        self.ceiling = chit.group(1) if chit else ''
        self.tool_names = _tool_names(tools)
        self.mask = self._build_mask()
        (self.root, self.warps, self.trace) = LoomPlan.weave(self.mask, self.variants)
        self.tension_unlocked = False

    def _build_mask(self) -> int:
        mask = 0
        joined = f'{self.shopper}\n{self.ops}'
        if len(self.variants) >= 2 or _RX_RANK.search(joined):
            mask |= BIT_CLOSED
        if _RX_ASK.search(self.shopper):
            mask |= BIT_ASK
        why = bool(_RX_WHY.search(self.shopper)) or bool(re.search('\\bwhy\\b|\\breason\\b|\\bjustif', self.shopper, re.I))
        if _RX_OPTIONAL_WHY.search(self.raw) and (not re.search('\\bwhy\\b|\\breason\\b|\\bjustif', self.shopper, re.I)):
            why = False
        if why:
            mask |= BIT_WHY
        if _RX_RECOVER.search(joined):
            mask |= BIT_RECOVER
        if _RX_FIRM_SOFT.search(joined):
            mask |= BIT_FIRM_SOFT
        if _RX_PREFER.search(joined):
            mask |= BIT_PREFER
        if self.quote or re.search('\\breformulate\\b|\\bshortlist\\b', joined, re.I):
            mask |= BIT_QUOTE
        if _RX_EXACT.search(joined) or mask & BIT_ASK:
            mask |= BIT_EXACT
        if _RX_MULTI.search(self.shopper):
            mask |= BIT_MULTI
        return mask

    def active(self) -> list[str]:
        return [self.root, *self.warps]

    def carries(self, strand: str) -> bool:
        return strand == self.root or strand in self.warps

    def raise_tension(self) -> bool:
        if self.tension_unlocked or self.carries(STRAND_RECAST):
            return False
        if _mutex_blocks(self.root, STRAND_RECAST):
            return False
        if len(self.warps) >= 2:
            self.warps[-1] = STRAND_RECAST
        else:
            self.warps.append(STRAND_RECAST)
        self.tension_unlocked = True
        self.trace.append('tension=>recast')
        return True

    def may_attach_claims(self) -> bool:
        return self.carries(STRAND_ATTEST) or bool(self.mask & BIT_WHY)

    def board_card(self) -> str:
        lines = ['# Latch board (public brief)']
        lines.append(f'- root strand: {self.root}')
        if self.warps:
            lines.append(f"- warp strands: {', '.join(self.warps)}")
        lines.append(f"- weave: {', '.join(self.trace)}")
        lines.append(f'- mask bits: {self.mask:#011b}')
        if self.ceiling:
            lines.append(f'- spend ceiling hint: {self.ceiling}')
        if self.quote:
            lines.append(f'- quoted listing phrase: {self.quote}')
        if self.variants:
            lines.append('- named variants: ' + ', '.join(self.variants))
        if self.mask & BIT_ASK:
            lines.append('- withheld firm figure: ask with message before search')
        if self.mask & BIT_WHY:
            lines.append('- order must carry grounded justification.claims')
        if self.tension_unlocked:
            lines.append('- tension unlock: RECAST warp added after market shock')
        return '\n'.join(lines)

class LoomPlan:

    @staticmethod
    def weave(mask: int, variants: list[str]) -> tuple[str, list[str], list[str]]:
        trace: list[str] = []
        work = mask
        if len(variants) >= 2:
            work |= BIT_CLOSED
        best_strand = STRAND_SCOUT
        best_score = -1
        best_req = 0
        for (required, strand) in _LATCH_PATTERNS:
            if required == 0:
                continue
            if work & required != required:
                continue
            score = _popcount(required)
            if score > best_score:
                best_score = score
                best_strand = strand
                best_req = required
        if best_score < 0:
            best_strand = STRAND_SCOUT
            best_req = 0
            trace.append('root<=default_scout')
        else:
            trace.append(f'root<={best_strand}@spec{best_score}')
        owned = _STRAND_OWNS.get(best_strand, 0) | best_req
        uncovered = work & ~owned
        warps: list[str] = []
        for (bit, strand) in _BIT_TO_WARP:
            if not uncovered & bit:
                continue
            if strand == best_strand:
                continue
            if _mutex_blocks(best_strand, strand):
                trace.append(f'skip_mutex:{strand}')
                continue
            if strand in warps:
                continue
            warps.append(strand)
            trace.append(f'warp<={strand}@bit{bit}')
            if len(warps) >= 2:
                break
        return (best_strand, warps, trace)
_PRIMER = "\nYou are BlackGirl v7. You purchase through a live EnvPack under a Latch Loom plan:\none root strand plus optional warp strands chosen from public problem signals.\n\nFinish when an observation sets done=true. Stay inside max_steps and\nmax_calls_per_turn. Call only tools listed in this episode's schemas.\nWhen max_calls_per_turn > 1, batch independent view/compare/inspect calls.\n\nBeat rhythm\n  PLUCK    issue search/filter using listing language (values, not schema keys).\n  TENSION  open concrete product_id+sku rows; inspect_stock when uncertain.\n  KNOT     cart the intended variant — the first cart line is scored.\n  TRIM     place_test_order after a fresh price/stock check. Do not loop\n           inspect_stock once the cart is verified — close the episode.\n\nTreat budget, stock, product class and named hard specs as walls.\nTreat brand leanings and comfort qualities as preferences that never break walls.\n\nWhen results are thin, change one query lever once (drop filler words or add a\ncategory filter). Do not thrash the search box.\n\nIf the shopper speaks a question, answer with message before touching cart/order.\n".strip()
_ROOT_BOOK: dict[str, str] = {STRAND_BRAID: '\n# Root BRAID · TF1 · donors minion_king, Snakehead\nSeparate walls from wishes. Search until you can view exact options that clear\nevery wall. Prefer the named maker on the first cart only when that option also\nclears every wall; otherwise take the strongest wall-clearing option and move on.\nIf the maker wish is impossible under the walls, do not freeze — buy any fully\ncompliant unit. Re-check stock, then order without optional claims. Reply to\nshopper interventions with message first.\n'.strip(), STRAND_SCOUT: "\n# Root SCOUT · TF2 · donors Victory ORO, PrecisionCart\nStart from the shopper's distinctive tokens. If the page is empty or off-class,\nperform exactly one near-miss refine. Collect several different product identities\nbefore committing. A quoted phrase must appear in the viewed title tokens; a brand\nhit alone is insufficient. Drop accessories and off-path categories. When two\nfinalists disagree on a wall, keep the one that satisfies more literal gates.\nOrder as soon as an in-budget, in-stock winner is verified.\n".strip(), STRAND_CLAMP: '\n# Root CLAMP · TF3 · donors New tttop, Jaguar\nBind every wall to the ordered option row — never to the parent title. View the\nsku, then inspect_stock on that sku. When the brief says a firm figure appears\nonly after you ask, your first action is message; search waits. Before ordering,\nre-read numbers and codes. If the chosen sku sells out, switch to a different\nproduct_id that still clears every wall; sibling options that miss a figure are\nnot acceptable. Treat named figures as written; do not invent class or unit\nconversions.\n'.strip(), STRAND_SAVOR: '\n# Root SAVOR · TF4 · donors Lion-King, pig1004\nWalls: product class, spend ceiling, availability, a real sellable unit.\nWishes: durability, materials, environment fit, connectivity, handling.\nBefore the first cart, compare at least two credible candidates on those wishes.\nA slightly higher price that still fits the ceiling can win when the fit is better.\nDo not cart a cheap stand-in. Soft adjectives alone should not reject a candidate\nwhose structured attributes already match.\n'.strip(), STRAND_LADDER: '\n# Root LADDER · TF5 · donor Superlovely\nWork inside the named product_id::sku set. Batch-view those ids on the first\nturn; avoid open search. Sort only by the priority axes the shopper listed, in\nthat order. First cart = current #1. After any stock, price or cart shock:\n  1) re-inspect remaining named options\n  2) rebuild the ranking from scratch (same axes)\n  3) remove the old cart line if it is no longer #1\n  4) cart the NEW leader, inspect_stock, place_test_order immediately\nNever keep a pre-shock winner that is now dominated. Never spend leftover\nsteps re-inspecting without ordering. place_test_order is product_id + sku\nonly — omit justification/claims.\n'.strip(), STRAND_RECAST: '\n# Root RECAST · TF6 · donors Staystaystay, Tenth\nWalls that never move: product class, spend ceiling, availability.\nWhen the target disappears, concede in this order only:\n  same model, other colour  →  different model that still clears walls.\nTrust an explicit model attribute over a marketing title that conflicts.\nWiden by dropping soft filters before touching walls. Cap hard swaps, then order.\n'.strip(), STRAND_ATTEST: "\n# Root ATTEST · TF7\nThe shopper asked why. A correct product with an empty order payload still\nscores zero. Before the first cart, view/compare the candidate's raw facts.\nAfter cart, inspect_stock, then place_test_order in the same episode.\n\nplace_test_order MUST carry justification.claims as field/value pairs:\n  - current price\n  - current in_stock\n  - the spec the shopper actually named, using the catalog attribute key\n    from the viewed facts (not a marketing phrase, not a title fragment)\n  - at most one extra evidenced reason (prefer brand when present)\n\nCopy values exactly as viewed/inspected. Do not invent features, rewrite\nunits, or omit the claims block. Do not keep sending message once you have\nanswered; finish the purchase.\n".strip()}
_WARP_NOTE: dict[str, str] = {STRAND_BRAID: '# Warp BRAID\nPreserve wall/wish separation; honour the maker wish on first cart only when every wall is already met.', STRAND_SCOUT: '# Warp SCOUT\nRequire a multi-identity shortlist; allow one thin near-miss refine.', STRAND_CLAMP: '# Warp CLAMP\nFinal option must pass an exact-sku gate; ask for withheld figures first when the brief requires it.', STRAND_SAVOR: '# Warp SAVOR\nAmong wall-equal options, choose on use-case fit rather than lowest price.', STRAND_LADDER: '# Warp LADDER\nNamed variants form a closed set; rebuild order after market shocks.', STRAND_RECAST: '# Warp RECAST\nAfter stock/price failure, climb the colour-then-model concession chain without loosening walls.', STRAND_ATTEST: '# Warp ATTEST\nIf the shopper asked why, attach grounded catalog claims on the closing order. Values must equal observed text.'}
_BEAT_GUIDE = '\n# Beat pacing\n- Opening: PLUCK then TENSION; ask gated figures immediately when required.\n- Middle: KNOT once evidence is enough — do not stockpile unused steps.\n- Closing (≥ ~70% of the step budget): stop hunting; TRIM the verified cart.\n- Reserve one or two turns after shocks for inspect_stock + place_test_order.\n'.strip()
_TIP_ASK = 'A firm figure is withheld until you ask. Send message now to collect it before any search.'
_TIP_REPLY = 'The shopper asked a question. Reply with message on this turn before any further cart or order action.'
_TIP_RERANK = 'The market moved — the previous leader is no longer the ranking. Re-inspect the remaining named variants, rebuild the order by the stated priority, remove the old cart line, add the NEW #1, inspect_stock, then place_test_order with product_id and sku only. Do not keep the pre-event pick. Do not loop inspect_stock.'
_TIP_CLOSE_LADDER = 'Stop exploring and stop re-inspecting. Ensure the current ranking leader is in the cart and place_test_order NOW with product_id and sku only (no justification).'
_TIP_CLAIMS = 'On place_test_order you MUST attach justification.claims: catalog keys from the goal and from viewed facts — price, in_stock, the asked spec, and at most one evidenced extra reason (prefer brand). No category, title, summary, or evidence labels. Values must equal observed catalog text; do not invent features or rewrite measurements. An order without claims fails even when the product is correct.'
_TIP_ATTEST_CLOSE = 'Candidate is carted. inspect_stock if the price/stock is stale, then place_test_order THIS turn with justification.claims (price, in_stock, asked spec, optional brand). Do not send another message.'
_TIP_STOP_TALK = 'You already answered. Stop calling message. View or inspect a real sellable unit, cart it, then place_test_order.'
_TIP_RECAST = 'Something failed or moved ({detail}). Recover with the smallest concession that still holds product class, spend ceiling and availability.'
_TIP_BRAID = 'First cart should take the stated leaning when a fully compliant sku offers it; otherwise take the best firm-compliant item. Answer questions with message.'
_TIP_COMPARE = 'Compare candidates on the use-case qualities the shopper implied before the first cart entry — price alone is not the objective.'
_TIP_REFRAME = 'Adjust one part of the query or add a filter so the shortlist spans several distinct products before you buy.'
_TIP_DEADLINE = "Only {left} step(s) remain. Confirm the current pick's price and stock, then place_test_order."
_TIP_TENSION = 'Market shock raised loom tension — RECAST warp is now active. Climb the least-relax ladder; do not keep a dominated pre-event pick.'

class ClaimWeft:

    @staticmethod
    def canon_field(raw: Any) -> str:
        label = str(raw or '').strip()
        if not label or label.lower() in _FIELD_DENY:
            return ''
        return label

    @staticmethod
    def tidy_value(field: str, raw: Any) -> str:
        text = '' if raw is None else str(raw).strip()
        if field.lower() == 'in_stock':
            return 'true' if text.lower() in {'true', '1', 'yes', 'in_stock'} else text
        return text

    @staticmethod
    def polish(claims: Any, board: LatchBoard) -> list[dict[str, str]]:
        del board
        if not isinstance(claims, list):
            return []
        kept: list[dict[str, str]] = []
        used: set[str] = set()
        for entry in claims:
            if not isinstance(entry, dict):
                continue
            field = ClaimWeft.canon_field(entry.get('field'))
            if not field:
                continue
            value = ClaimWeft.tidy_value(field, entry.get('value'))
            if not value:
                continue
            key = field.lower()
            if key in _FIELD_DENY or key in used:
                continue
            used.add(key)
            kept.append({'field': field, 'value': value})
        return kept

def _sanitize_model(model_id: str) -> str:
    mid = (model_id or '').strip()
    if not mid:
        return _MODEL_LIGHT
    mapped = _CHUTES_TO_OR.get(mid, mid)
    low = mapped.lower()
    if 'kimi' in low or 'glm' in low or low.startswith('z-ai/') or low.startswith('moonshotai/'):
        return _MODEL_DENSE
    return mapped

def _pick_model(board: LatchBoard) -> str:
    override = (getenv('SANDBOX_MODEL') or '').strip()
    if override:
        return _sanitize_model(override)
    if any((s in _DENSE_STRANDS for s in board.active())):
        return _MODEL_DENSE
    return _MODEL_LIGHT

def _compose_system(board: LatchBoard, policy: dict[str, Any]) -> str:
    parts = [_PRIMER, _ROOT_BOOK.get(board.root, _ROOT_BOOK[STRAND_SCOUT])]
    for warp in board.warps:
        parts.append(_WARP_NOTE.get(warp, ''))
    parts.append(board.board_card())
    parts.append(_BEAT_GUIDE)
    tools = ', '.join(board.tool_names) if board.tool_names else '(see schemas)'
    parts.append(f"# Runtime\nAvailable tools: {tools}\nmax_steps={policy.get('max_steps')} max_calls_per_turn={policy.get('max_calls_per_turn', 1)}")
    return '\n\n'.join((p for p in parts if p))

def _load_args(tool_call: dict[str, Any]) -> dict[str, Any]:
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

def _shrink(obs: Any, limit: int=_OBS_CEILING) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    if len(text) <= limit:
        return text
    return text[:limit - 16] + '...<truncated>'

def _names_of(tool_calls: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for call in tool_calls:
        fn = call.get('function') if isinstance(call, dict) else None
        if isinstance(fn, dict) and fn.get('name'):
            out.append(str(fn['name']))
    return out

def _is_question(text: str) -> bool:
    body = (text or '').strip().lower()
    if not body:
        return False
    if '?' in body:
        return True
    openers = ('which', 'what', 'does', 'did', 'can you', 'could you', 'would you', 'wait', 'hold on', 'before you', 're-check', 'recheck')
    return any((body.startswith(word) for word in openers))

def _market_shifted(observation: Any) -> bool:
    if not isinstance(observation, dict):
        return False
    for key in ('market_event', 'stock_changed', 'price_changed', 'event'):
        if observation.get(key):
            return True
    nested = observation.get('observation')
    if isinstance(nested, dict):
        for key in ('market_event', 'stock_changed', 'price_changed'):
            if nested.get(key):
                return True
    return False

def _ingest_catalog(store: dict[str, dict[str, Any]], observation: Any) -> None:
    if not isinstance(observation, dict):
        return
    pid = str(observation.get('product_id') or '').strip()
    base: dict[str, Any] = {}
    for key in ('brand', 'price', 'in_stock'):
        if observation.get(key) not in (None, ''):
            base[key] = observation[key]
    facts = observation.get('facts')
    if isinstance(facts, dict):
        base['facts'] = dict(facts)
    sku_top = str(observation.get('sku') or '').strip()
    if pid:
        store[pid] = {**store.get(pid, {}), **base}
        if sku_top:
            store[f'{pid}::{sku_top}'] = {**store.get(f'{pid}::{sku_top}', {}), **base}
    variants = observation.get('variants')
    if not isinstance(variants, list):
        return
    for row in variants:
        if not isinstance(row, dict):
            continue
        sku = str(row.get('sku') or '').strip()
        if not pid or not sku:
            continue
        piece = dict(base)
        if row.get('price') not in (None, ''):
            piece['price'] = row['price']
        if 'in_stock' in row:
            piece['in_stock'] = row['in_stock']
        store[f'{pid}::{sku}'] = {**store.get(f'{pid}::{sku}', {}), **piece}

def _ground_price_stock(store: dict[str, dict[str, Any]], pid: str, sku: str) -> list[dict[str, str]]:
    rec: dict[str, Any] = {}
    if pid and sku:
        rec = store.get(f'{pid}::{sku}') or {}
    if not rec:
        rec = store.get(pid) or {}
    out: list[dict[str, str]] = []
    for field in ('price', 'in_stock', 'brand'):
        value = rec.get(field)
        if value in (None, ''):
            continue
        text = str(value).strip()
        if field == 'in_stock':
            text = 'true' if text.lower() in {'true', '1', 'yes', 'in_stock'} else text
        out.append({'field': field, 'value': text})
    return out

def _fault_text(observations: list[Any]) -> str:
    for item in observations:
        if isinstance(item, dict) and item.get('error'):
            return str(item['error'])
    return ''

class BeatClock:
    __slots__ = ('turn', 'limit', 'last_acts', 'ever', 'seen', 'knotted', 'trimmed', 'owes_reply', 'market_moved', 'asked_spec', 'coached', 'cart_pid', 'cart_sku', 'talks', 'catalog', 'market_turn')

    def __init__(self, limit: int) -> None:
        self.turn = 0
        self.limit = limit
        self.last_acts: list[str] = []
        self.ever: set[str] = set()
        self.seen: list[Any] = []
        self.knotted = False
        self.trimmed = False
        self.owes_reply = False
        self.market_moved = False
        self.asked_spec = False
        self.coached: set[str] = set()
        self.cart_pid = ''
        self.cart_sku = ''
        self.talks = 0
        self.catalog: dict[str, dict[str, Any]] = {}
        self.market_turn = 0

    @property
    def left(self) -> int:
        return max(0, self.limit - self.turn)

    def note(self, labels: list[str]) -> None:
        self.last_acts = labels
        self.ever.update(labels)
        if 'add_to_cart' in labels:
            self.knotted = True
        if 'place_test_order' in labels:
            self.trimmed = True
        if 'message' in labels:
            self.owes_reply = False
            self.asked_spec = True
            self.talks += 1

    def once(self, key: str) -> bool:
        if key in self.coached:
            return False
        self.coached.add(key)
        return True

    def beat(self) -> str:
        if self.trimmed:
            return BEAT_TRIM
        if self.knotted:
            return BEAT_KNOT
        if self.ever & {'view', 'compare', 'inspect_stock'}:
            return BEAT_TENSION
        return BEAT_PLUCK

class LoomEpisode:

    def __init__(self, problem_data: dict[str, Any]) -> None:
        environment = problem_data['environment']
        self.binding = environment['binding']
        self.policy = environment['policy_view']
        self.schemas = self.policy.get('tools') or []
        self.brief = str(self.policy.get('query') or problem_data.get('query') or '')
        self.key = str(problem_data.get('problem_id', problem_data.get('id', 'problem')))
        self.per_turn = max(1, int(self.policy.get('max_calls_per_turn', 1)))
        self.board = LatchBoard(self.brief, self.schemas)
        self.clock = BeatClock(max(1, int(self.policy.get('max_steps', 20))))
        self.model = _pick_model(self.board)

    def _opening(self) -> list[dict[str, Any]]:
        transcript: list[dict[str, Any]] = [{'role': 'system', 'content': _compose_system(self.board, self.policy)}, {'role': 'user', 'content': self.brief}]
        if self.board.carries(STRAND_CLAMP) and self.board.mask & BIT_ASK:
            transcript.append({'role': 'user', 'content': _TIP_ASK})
        elif self.board.carries(STRAND_ATTEST):
            transcript.append({'role': 'user', 'content': _TIP_CLAIMS})
        return transcript

    def _coach(self) -> str | None:
        clk = self.clock
        board = self.board
        turn = clk.turn
        if clk.owes_reply and clk.talks < 2:
            return _TIP_REPLY
        if clk.owes_reply and clk.talks >= 2 and clk.knotted:
            return _TIP_ATTEST_CLOSE if board.carries(STRAND_ATTEST) else _TIP_CLOSE_LADDER if board.carries(STRAND_LADDER) else _TIP_DEADLINE.format(left=clk.left)
        if board.carries(STRAND_CLAMP) and turn == 1 and board.mask & BIT_ASK and (not clk.asked_spec):
            return _TIP_ASK
        if board.carries(STRAND_LADDER) and (not clk.trimmed):
            named = ', '.join(board.variants) if board.variants else 'the named variants'
            if clk.market_moved:
                since = clk.turn - clk.market_turn
                if since >= 3 or clk.left <= 6:
                    return _TIP_CLOSE_LADDER
                return _TIP_RERANK + f' Named set: {named}.'
            if clk.left <= 3:
                return _TIP_CLOSE_LADDER
            if turn == 1:
                return f'Batch-view these variants now: {named}. Sort only by the stated priority, first-cart current #1. After any market move, rebuild the ranking and order the new leader with product_id and sku only — no justification.'
        if turn == 1 and board.carries(STRAND_BRAID):
            return _TIP_BRAID
        if clk.talks >= 2 and (not clk.owes_reply) and (not clk.knotted) and clk.once('stop_talk'):
            return _TIP_STOP_TALK
        if turn <= 2 and (not clk.ever & {'search', 'filter', 'view', 'compare', 'message'}):
            return "Begin with search or filter using words from the shopper's own sentence, then open the best hits."
        if board.carries(STRAND_ATTEST) and (not clk.trimmed):
            if clk.knotted:
                return _TIP_ATTEST_CLOSE
            if turn in (1, 2, 5) or clk.left <= 4:
                return _TIP_CLAIMS
        if board.carries(STRAND_RECAST):
            detail = _fault_text(clk.seen)
            if detail:
                return _TIP_RECAST.format(detail=detail)
        if board.carries(STRAND_SCOUT) and turn == 3 and (not clk.knotted) and ('search' in clk.ever):
            return _TIP_REFRAME
        if board.carries(STRAND_SAVOR) and turn == 2 and ('compare' not in clk.ever):
            return _TIP_COMPARE
        if not clk.trimmed and clk.knotted and (clk.left <= 2) and clk.once('deadline'):
            return _TIP_DEADLINE.format(left=clk.left)
        return None

    def _scrub_order(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        payload = dict(args)
        clk = self.clock
        if name == 'add_to_cart':
            clk.cart_pid = str(payload.get('product_id') or '')
            clk.cart_sku = str(payload.get('sku') or '')
            return payload
        if name == 'remove_from_cart':
            if str(payload.get('product_id') or '') == clk.cart_pid:
                clk.cart_pid = ''
                clk.cart_sku = ''
            return payload
        if name != 'place_test_order':
            return payload
        if not payload.get('product_id') and clk.cart_pid:
            payload['product_id'] = clk.cart_pid
            if clk.cart_sku:
                payload['sku'] = clk.cart_sku
        if not self.board.may_attach_claims():
            payload.pop('justification', None)
            return payload
        note = payload.get('justification')
        raw = note.get('claims') if isinstance(note, dict) else None
        kept = ClaimWeft.polish(raw, self.board)
        if not kept:
            kept = ClaimWeft.polish(_ground_price_stock(clk.catalog, str(payload.get('product_id') or ''), str(payload.get('sku') or '')), self.board)
        payload['justification'] = {'claims': kept}
        return payload

    def _infer(self, transcript: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        ladder = [self.model]
        for spare in _MODEL_FALLBACK:
            if spare not in ladder:
                ladder.append(spare)
        trouble: Exception | None = None
        for candidate in ladder[:3]:
            payload = _proxy.post('/inference/chat/completions', json_data={'model': candidate, 'messages': transcript, 'tools': self.schemas, 'tool_choice': 'required', 'temperature': 0})
            if payload is None:
                trouble = RuntimeError(f'no completion from {candidate}')
                continue
            try:
                reply = payload['choices'][0]['message']
            except (KeyError, IndexError, TypeError) as exc:
                trouble = exc
                continue
            spoken = reply.get('content') or ''
            calls = list(reply.get('tool_calls') or [])[:self.per_turn]
            return (spoken, calls)
        raise RuntimeError(f'inference request failed ({trouble})')

    def _dispatch(self, calls: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        batch = f'{self.key}-turn-{turn}'
        actions = []
        for (slot, call) in enumerate(calls, start=1):
            fn = call.get('function') if isinstance(call, dict) else None
            name = str(fn.get('name') or '') if isinstance(fn, dict) else ''
            if not name:
                raise RuntimeError('model produced a tool call with no name')
            args = self._scrub_order(name, _load_args(call))
            if isinstance(fn, dict):
                fn['arguments'] = json.dumps(args)
            actions.append({'call_id': f'{batch}-{slot}', 'action': {'name': name, 'args': args}})
        reply = _proxy.post('/environment/call', json_data={**self.binding, 'call_id': batch, 'idempotency_key': batch, 'turn': turn, 'calls': actions})
        if reply is None:
            raise RuntimeError('environment call failed')
        return reply

    def _speak(self, transcript: list[dict[str, Any]], trail: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        for attempt in range(len(_IDLE_NUDGES) + 1):
            (spoken, calls) = self._infer(transcript)
            if calls:
                return (spoken, calls)
            trail.append({'role': 'assistant', 'content': spoken})
            transcript.extend([{'role': 'assistant', 'content': spoken}, {'role': 'user', 'content': _IDLE_NUDGES[min(attempt, len(_IDLE_NUDGES) - 1)]}])
        raise RuntimeError('engine would not emit an executable action')

    def _absorb(self, result: dict[str, Any], transcript: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
        clk = self.clock
        rows = result.get('calls') or []
        clk.seen = []
        for (index, call) in enumerate(calls, start=1):
            row = rows[index - 1] if index - 1 < len(rows) else None
            observation = row.get('observation') if isinstance(row, dict) else None
            clk.seen.append(observation)
            _ingest_catalog(clk.catalog, observation)
            call_id = call.get('id') if isinstance(call, dict) else None
            transcript.append({'role': 'tool', 'tool_call_id': str(call_id) if call_id else f'{self.key}-{clk.turn}-{index}', 'content': _shrink(observation)})
            if _market_shifted(observation):
                if not clk.market_moved:
                    clk.market_turn = clk.turn
                clk.market_moved = True
        note = result.get('user_message')
        if isinstance(note, dict) and note.get('content'):
            spoken = str(note['content'])
            transcript.append({'role': 'user', 'content': spoken})
            if _is_question(spoken):
                clk.owes_reply = True
            if self.board.carries(STRAND_LADDER) and clk.knotted:
                if not clk.market_moved:
                    clk.market_turn = clk.turn
                clk.market_moved = True
        if clk.market_moved and self.board.raise_tension():
            transcript.append({'role': 'user', 'content': _TIP_TENSION})
        elif self.board.carries(STRAND_RECAST):
            detail = _fault_text(clk.seen)
            if detail:
                transcript.append({'role': 'user', 'content': _TIP_RECAST.format(detail=detail)})
        elif self.board.carries(STRAND_LADDER) and clk.market_moved and (not clk.trimmed) and clk.once('absorb_rerank'):
            transcript.append({'role': 'user', 'content': _TIP_RERANK})

    @staticmethod
    def _done(result: dict[str, Any]) -> bool:
        for row in result.get('calls') or []:
            if not isinstance(row, dict):
                continue
            observation = row.get('observation')
            if isinstance(observation, dict) and observation.get('done'):
                return True
        return False

    def run(self) -> list[dict[str, Any]]:
        transcript = self._opening()
        trail: list[dict[str, Any]] = [{'role': 'meta', 'blackgirl': {'version': 7, 'root': self.board.root, 'warps': list(self.board.warps), 'model': self.model, 'provider': 'openrouter', 'mixer': 'latch_loom'}}]
        clk = self.clock
        while clk.turn < clk.limit:
            clk.turn += 1
            tip = self._coach()
            if tip:
                transcript.append({'role': 'user', 'content': tip})
            (spoken, calls) = self._speak(transcript, trail)
            transcript.append({'role': 'assistant', 'content': spoken, 'tool_calls': calls})
            result = self._dispatch(calls, clk.turn)
            trail.append({'role': 'assistant', 'content': spoken, 'tool_calls': calls, 'environment_result': result, 'blackgirl_turn': {'root': self.board.root, 'warps': list(self.board.warps), 'beat': clk.beat(), 'turn': clk.turn}})
            clk.note(_names_of(calls))
            self._absorb(result, transcript, calls)
            if self._done(result):
                break
        return trail

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return LoomEpisode(problem_data).run()
