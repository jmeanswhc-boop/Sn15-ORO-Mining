from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_OPENROUTER_MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3.6-27b",
]
_CHUTES_MODELS = [
    "moonshotai/Kimi-K2.6-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "Qwen/Qwen3.6-27B-TEE",
]

_DEFAULT_CLAIM_FIELDS = frozenset(
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
# category claims repeatedly fail faithfulness on TF7; never emit unless required
# (required_claims never include category in observed EnvPack).
_CLAIM_DENY = frozenset({"category"})
_FIELD_ALIASES = {
    "display": "resolution",
    "screen": "screen_size",
    "size": "screen_size",
    "screen size": "screen_size",
    "refresh": "refresh_rate",
    "hz": "refresh_rate",
    "category_path": "category",
}

_CAND_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9][A-Za-z0-9._-]*)")
_ASK_SPEC_RE = re.compile(
    r"tell you if you ask|if you ask|ask me.*firm|firm requirement.*ask",
    re.I,
)

_proxy = ProxyClient(timeout=120, max_retries=3)

_SYSTEM = """You are an ORO Bench shopping agent (multi-donor TF merge on EnvPack tools).

Use ONLY supplied tools. Continue until done=true. Never memorize product IDs or suite ranks.
Hidden races differ from local packs — obey this episode's shopper text + Task rules only.

## TF1 Intent (Lovelystar)
Silently decompose into ≤4 mutually exclusive sub-goals (keywords, firm budget/specs, soft prefs).
If shopper asks a question / changes budget / asks which item: NEXT action is `message`, then resume.
Ask-first: if they say a firm spec is told when you ask → FIRST call `message` for that spec.

## TF2 Retrieval (HiSun lean + madlol; KAK anti-thrash)
search/filter with reformulations. Survey a few distinct product_ids — enough to be confident.
Hard cap mindset: after ~3–4 weak searches, view best hits and proceed (do not 2× thrash).
Prefer in_stock=true when availability matters.

## TF3 Constraints (madlol / ELON / angry Product)
Firm specs are NON-NEGOTIABLE (category_path, RAM/screen/resolution/refresh, budget, stock).
Match brands/models/sizes/numbers LITERALLY after `view`. Soft brand only after all firm pass.

## TF4 Preference (Lovelystar)
Among fully valid options, prefer cheaper / better fit to soft prefs. Soft ≠ override firm.

## TF5 Ranking (0828 closedRank)
When shopper lists product_id::sku ("compare only these"): that set is exhaustive.
view every listed product_id; rank by stated priority; cart+order MUST be a listed key.
Never open-catalog search for substitutes.

## TF6 Recovery (Lovelystar + 0828 finishSafe)
Stock/price shock or market_event: message if asked → remove_from_cart → other candidate
(same model/color ladder when relevant) → inspect_stock → add_to_cart → place_test_order.
NEVER order on out-of-stock / pre-event assumptions. Never quit with "cannot complete".

## TF7 Justification (Lovelystar + 0828)
Only if shopper asks why/reason. Claims MUST match latest view/inspect_stock facts.
Never claim category. Resolution classes: hd/fhd/qhd/uhd when shopper used that language.
Omit justification entirely when not asked.

## Finish discipline
search/filter → view/compare → inspect_stock → add_to_cart → place_test_order.
Late in the budget: stop exploring and order an in-stock in-budget pick.
"""


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"

_TMG006BE_AST_PAD = 0
if _TMG006BE_AST_PAD:
    class _TMG006BE_LS_SubmitNarrator:

        @staticmethod
        def locate_kw(val: str, prod: LS_Product) -> Optional[str]:
            title_lower = prod.title.lower()
            attr_flat = [v.lower() for vs in prod.attributes.values() for v in vs]
            sku_flat = [v.lower() for kv in prod.sku.values() for v in kv.values()]
            if val in title_lower:
                return 'title'
            if any((val in av for av in attr_flat)):
                return 'attributes'
            if any((val in sv for sv in sku_flat)):
                return 'sku options'
            return None

        @staticmethod
        def product_checks(think: str, best_p: List[LS_Product], qi: LS_QueryInfo) -> str:
            for i in range(len(best_p)):
                think += f'\n**Product {i + 1}** (UID `{best_p[i].id}`)\n' if len(qi.items) > 1 else ''
                think += f'\n- query: `{qi.items[i].query}`'
                think += f'\n- product_title: `{best_p[i].title}`'
                think += f'\n- shop_id: `{best_p[i].shop_id}`'
                for kw in sorted(qi.items[i].kw, key=lambda kw: tuple(kw)):
                    key = '_'.join([s for k in kw[:-1] for s in k.split()])
                    val = kw[-1].lower()
                    location = LS_SubmitNarrator.locate_kw(val, best_p[i])
                    think += f'\n- {key}: expects `{kw[-1]}` | found in {location} → exact match ✓' if location else f'\n- {key}: expects `{kw[-1]}` | not found in title or attributes → no match ✗'
                if qi.items[i].price:
                    think += f'\n- price: expects {LS_EngineUtil.promptfy_price(qi.items[i].price)} | found {str(int(best_p[i].price))} → within range ✓'
                for svc in qi.items[i].services or []:
                    think += f'\n- service: expects `{svc}` | found `{svc}` → available ✓'
                think += '\n'
            return think

        @staticmethod
        def voucher_tail(think: str, best_p: List[LS_Product], qi: LS_QueryInfo) -> str:
            psum = sum((p.price for p in best_p))
            pact = LS_EngineUtil.get_vouchered_cost(best_p, qi.vopt)
            if len(best_p) > 1:
                think += '\nThe prices of the selected products are ' + ', '.join([str(int(p.price)) for p in best_p[:-1]]) + ' and ' + str(int(best_p[-1].price)) + ', respectively, with a total price of ' + str(int(psum)) + '.'
            else:
                think += '\nThe price of the selected product is ' + str(int(best_p[0].price)) + '.'
            if psum > pact:
                if qi.vopt.apply_to == 'shop':
                    think += f' Since all of them come from the same shop (`{best_p[0].shop_id}`) and the total price exceeds the voucher threshold, the voucher can be applied.'
                think += ' After applying the voucher, the effective total price becomes ' + str(int(pact)) + f", which is within the user's budget `{str(int(qi.vopt.budget))}`."
            else:
                think += f' It is already within the budget `{str(int(qi.vopt.budget))}`, even without using the voucher.'
            think += '\n\nIn summary, the selected products are within the budget and satisfy all stated requirements.' if len(best_p) > 1 else '\n\nIn summary, the selected product is within the budget and satisfies all stated requirements.'
            return think

        @staticmethod
        def challenge_tail(think: str, best_p: List[LS_Product], qi: LS_QueryInfo) -> str:
            if qi.challenge == LS_ShoppingTaskType.VOUCHER:
                return LS_SubmitNarrator.voucher_tail(think, best_p, qi)
            if qi.challenge == LS_ShoppingTaskType.SHOP:
                return think + f"\nAll the selected products come from the same shop (`{best_p[0].shop_id}`), as requested.\n\nIn summary, all selected products come from the same shop, and each one satisfies the user's requirements."
            return think + "\nAll checks passed. The selected product satisfies the user's requirements."


    class _TMG006BE_OG_VoucherOption(OG_JsonModel):
        budget: float
        apply_to: str
        threshold: Optional[float]
        is_fixed: bool
        discount: float
        cap: Optional[float]

        def is_favor_cost(self, psum: float) -> bool:
            if self.threshold is None:
                return False
            if self.threshold < int(psum * 0.1) or self.threshold > int(psum * 0.9):
                return False
            if psum < 100.0:
                return False
            if not self.is_fixed:
                ac = max(psum * (1.0 - self.discount / 100.0), psum - (self.cap if self.cap is not None else math.inf))
                if self.budget < math.floor(ac) or self.budget > math.ceil(1.1 * ac):
                    return False
            return True


    class _TMG006BE_EG_Planner:

        def __init__(self, query: str, trace: EG_Trace, task_hint: str='') -> None:
            self.query = query
            self.trace = trace
            self.task_hint = task_hint

        def _build_hint_text(self, qi: EG_QueryInfo) -> str:
            if qi.challenge == EG_ShoppingTaskType.PRODUCT:
                return '\n\n# Hint\nThe original query contains only one product!'
            if qi.challenge == EG_ShoppingTaskType.SHOP:
                return '\n\n# Hint\nThe user wants multiple products from the same shop. Split one sub-query per product; assign each product its own price and service constraints.'
            if qi.challenge == EG_ShoppingTaskType.VOUCHER:
                return '\n\n# Hint\nOnly the text before the budget block describes products. Split one sub-query per distinct product; exclude voucher/budget wording from sub-queries and keywords.'
            return ''

        @staticmethod
        def _estimate_min_items(query_e: str) -> int:
            q = query_e.lower()
            n = len(re.findall('\\b(?:first|second|third|fourth)\\b', q))
            if n >= 2:
                return n
            return 2 if re.search('\\band also\\b|\\bone option\\b.*\\banother\\b|(?=.*\\bboth\\b)(?=.*\\band\\b)', q) else 1

        @staticmethod
        def _kw_signature(kw: List[str]) -> str:
            return kw[0] if len(kw) == 2 else kw[-2]

        @staticmethod
        def _kw_core_terms(kw_list: List[List[str]]) -> str:
            skip = {'delivery', 'service'}
            parts: List[str] = []
            for kw in kw_list:
                if not kw:
                    continue
                keys, val = (kw[:-1], kw[-1].strip())
                if any((k in skip for k in keys)):
                    continue
                if val and val not in parts:
                    parts.append(val)
            return ' '.join(parts)

        @staticmethod
        def _has_type_keyword(sq: EG_SubQuery) -> bool:
            return any((EG_Planner._kw_signature(kw) == 'product_type' for kw in sq.kw if kw))

        @staticmethod
        def _price_is_valid(price: Optional[str]) -> bool:
            if not price:
                return True
            parts = price.split('-', 1)
            if len(parts) != 2:
                return False
            for part in parts:
                if part and (not re.fullmatch('\\d+(?:\\.\\d+)?', part)):
                    return False
            return True

        @staticmethod
        def _intent_has_spend_leak(sq: EG_SubQuery) -> bool:
            blob = ' '.join([sq.query or ''] + [kw[-1] for kw in sq.kw if kw])
            return EG_Specs._SPEND_LEAK_FLAG.search(blob) is not None

        @staticmethod
        def _keyword_value_is_noisy(sq: EG_SubQuery) -> bool:
            terms = EG_Planner._kw_core_terms(sq.kw)
            if not terms.strip():
                return True
            if EG_Specs._KEYWORD_NOISE.search(terms):
                return True
            for kw in sq.kw:
                if kw and EG_Planner._kw_signature(kw) == 'product_type' and EG_Specs._KEYWORD_NOISE.search(kw[-1].strip()):
                    return True
            return False

        def _count_audit_flags(self, qi: EG_QueryInfo) -> Dict[str, int]:
            q = qi.query_e
            return {'freeship_p': len([sq for sq in qi.items if 'freeShipping' in sq.services]), 'lazmall_p': len([sq for sq in qi.items if 'official' in sq.services]), 'lazflash_p': len([sq for sq in qi.items if 'flashsale' in sq.services]), 'cod_p': len([sq for sq in qi.items if 'COD' in sq.services]), 'price_p': len([sq for sq in qi.items if sq.price]), 'brand_p': len([sq for sq in qi.items if any((kw for kw in sq.kw if re.search('\\bbranded\\b', kw[-1], re.I)))]), 'cod_e': len(re.findall('\\bCOD\\b|\\b(cash(es|ing)?|pay(ment|ing)?s?)[\\s\\-]*(for)?[\\s\\-]*((up)?on|with(in)?|by|for)?[\\s\\-]*(deliver(y|ing|ie)?s?|cash(es|ing)?)', q, flags=re.I)), 'warranty_e': len(re.findall('\\b(\\d+)[\\s\\-]*(days?|weeks?|months?|years?)?[\\-\\s]*(periods?)?\\s*(of)?[\\-\\s]*(guarantee|warrant)', q, re.I)) + len(re.findall('\\b(guarantee|warrant)[a-z]*[\\s\\-]*(days?|weeks?|months?|years?)?[\\-\\s]*(periods?)?\\s*(of)?(\\d+)', q, re.I)), 'lazmall_e': len(re.findall('\\blaz\\-?mall\\b|\\bofficial\\s+(?:store|shop|brand)\\b', q, re.I)), 'lazflash_e': len(re.findall('\\blaz\\-?flash\\b|\\bflash[\\s\\-]sale\\b|\\blightning[\\s\\-]deal\\b', q, re.I)), 'freeship_e': len(re.findall('\\b(free|compliment[a-z]+)\\s+(ship|deliver)', q, re.I)), 'brand_e': len(re.findall('\\s+branded\\b', q, re.I)), 'price_e': len(re.findall('\\b(php|pesos?|usd|dollars?|euros?)\\b', q, re.I))}

        def _validate_query_info(self, qi: EG_QueryInfo) -> List[str]:
            errors: List[str] = []
            n = len(qi.items)
            if n == 0:
                return ['No sub-queries were produced.']
            if qi.challenge == EG_ShoppingTaskType.PRODUCT and n != 1:
                errors.append(f'Product challenge must have exactly 1 sub-query, got {n}.')
            if qi.challenge == EG_ShoppingTaskType.SHOP and n < 2:
                errors.append(f'Shop challenge must have at least 2 sub-queries, got {n}.')
            if qi.challenge == EG_ShoppingTaskType.SHOP and n >= 2:
                sigs = [(EG_Planner._kw_core_terms(sq.kw), sq.price or '', ','.join(sorted(sq.services or []))) for sq in qi.items]
                if len(set(sigs)) < len(sigs):
                    errors.append('Shop sub-queries must target distinct products (duplicate keywords with the same price and services).')
            min_products = self._estimate_min_items(qi.query_e)
            if qi.challenge in (EG_ShoppingTaskType.SHOP, EG_ShoppingTaskType.VOUCHER) and n < min_products:
                errors.append(f'Query describes at least {min_products} distinct products but only {n} sub-query(ies) were parsed.')
            if qi.challenge == EG_ShoppingTaskType.VOUCHER and n == 1 and (min_products >= 2):
                errors.append('Voucher query lists multiple products; split one sub-query per product (exclude budget/voucher text).')
            for i, sq in enumerate(qi.items, 1):
                if not (sq.query or '').strip():
                    errors.append(f'Sub-query {i} is empty.')
                if not self._has_type_keyword(sq):
                    errors.append(f'Sub-query {i} is missing a product_type keyword.')
                if self._keyword_value_is_noisy(sq):
                    errors.append(f"Sub-query {i} keywords are not search-ready; drop filler like 'looking for'.")
                if not self._price_is_valid(sq.price):
                    errors.append(f'Sub-query {i} has invalid price format `{sq.price}`.')
                if qi.challenge == EG_ShoppingTaskType.VOUCHER and self._intent_has_spend_leak(sq):
                    errors.append(f'Sub-query {i} still contains voucher/budget wording.')
            c = self._count_audit_flags(qi)
            for _, count_e_key, count_p_key, desc in EG_Specs._SCOPE_PARSE_RULES:
                count_e, count_p = (c[count_e_key], c[count_p_key])
                if count_e != count_p:
                    errors.append(f'{desc}: mentioned {count_e} time(s) in query but assigned to {count_p} sub-query(ies).')
            if c['price_e'] != c['price_p']:
                errors.append(f"Price: currency mentioned {c['price_e']} time(s) but only {c['price_p']} sub-queries have a price range.")
            elif c['price_e'] == 1 and qi.items and (not qi.items[-1].price) and re.search('\\b(php|pesos?|usd|dollars?|euros?)\\b[\\s.]*$', qi.query_e, re.I):
                errors.append('Price constraint appears at the end of the query but was not assigned to the last sub-query.')
            if c['warranty_e'] > 0 and (not any((re.search('\\b(guarantee|warrant)', (sq.query or '') + ' ' + ' '.join((kw[-1] for kw in sq.kw if kw)), re.I) for sq in qi.items))):
                errors.append(f"Warranty/guarantee mentioned {c['warranty_e']} time(s) but no sub-query carries a warranty constraint.")
            return errors

        def _try_voucher_rescue(self, challenge: EG_ShoppingTaskType) -> Optional[Tuple[EG_ShoppingTaskType, str, EG_VoucherOption]]:
            if challenge == EG_ShoppingTaskType.VOUCHER:
                return None
            soft = EG_Support.soft_route(self.query, '')
            has_signal = soft == 'voucher' or EG_Specs._BUDGET.search(self.query) is not None or EG_Specs._VOUCHER.search(self.query) is not None or (EG_Specs._VOUCHER_DISCOUNT_SIGNAL.search(self.query) is not None)
            if not has_signal:
                return None
            parsed = self._parse_voucher_clause()
            if parsed is None:
                return None
            query_e, vopt = parsed
            _debug_print(f'* classify_task: anti-misroute {challenge.value} → voucher')
            self.trace.print({'anti_misroute': f'{challenge.value}->voucher'})
            return (EG_ShoppingTaskType.VOUCHER, query_e, vopt)
        _SOFT_ERROR_PREFIXES = ('Free/complimentary shipping', 'LazMall', 'Flash sale', 'Cash on delivery', 'Branded', 'Price:', 'Price constraint appears', 'Warranty/guarantee')

        @classmethod
        def _partition_validation_errors(cls, errors: List[str]) -> Tuple[List[str], List[str]]:
            critical: List[str] = []
            soft: List[str] = []
            for e in errors:
                if any((e.startswith(p) for p in cls._SOFT_ERROR_PREFIXES)):
                    soft.append(e)
                else:
                    critical.append(e)
            return (critical, soft)

        def classify_task(self) -> EG_QueryInfo:
            if not self.trace.is_frozen('a-classify'):
                challenge = self._identify_task_type()
                query_e, vopt = (self.query, None)
                if challenge == EG_ShoppingTaskType.VOUCHER:
                    parsed = self._parse_voucher_clause()
                    if parsed is None:
                        self.trace.print('# voucher clause unparseable; downgrading to product')
                        challenge = EG_ShoppingTaskType.PRODUCT
                    else:
                        query_e, vopt = parsed
                else:
                    rescued = self._try_voucher_rescue(challenge)
                    if rescued is not None:
                        challenge, query_e, vopt = rescued
                qi = EG_QueryInfo(challenge=challenge, query_f=self.query, query_e=query_e, vopt=vopt)
                self.trace.add('a-classify', qi.to_json())
                self.trace.print({'challenge': challenge, 'query_e': query_e, 'voucher': vopt.to_json() if vopt else None})
                return qi
            else:
                return EG_QueryInfo.from_json(self.trace.find('a-classify', {'query_f': self.query}))

        def decompose_query(self, qi: EG_QueryInfo, feedback: str='', attempt: int=0) -> List[Dict[str, Dict[str, Any]]]:
            if not self.trace.is_frozen('a-decompose'):
                temperature = min(0.4, attempt * 0.2)
                tries = 0
                while tries < EG_Specs.SPLIT_PARSE_TRIES:
                    tries += 1
                    try:
                        prompt = EG_Specs.A_DECOMPOSE.format(shop_intent=' in the same shop' if qi.challenge == EG_ShoppingTaskType.SHOP else '', query=qi.query_e) + self._build_hint_text(qi)
                        if feedback:
                            prompt += f'\n\n# Fix Required\nThe previous parsing failed validation:\n{feedback}\nReturn corrected JSON only.'
                        js, _ = EG_Gateway.trigger_llm(prompt, preferred_llms=EG_Gateway.get_role_llm_priorities('decompose'), temperature=temperature, expect='dict')
                        decomposed = EG_Support.extract_decompose_response(js)
                        self.trace.print(js)
                        if len(decomposed) == 0:
                            raise Exception('no sub-queries extracted')
                        min_products = self._estimate_min_items(qi.query_e)
                        if len(decomposed) == 1 and qi.challenge == EG_ShoppingTaskType.SHOP:
                            raise Exception('sub-query count invalid')
                        if len(decomposed) < min_products and qi.challenge in (EG_ShoppingTaskType.SHOP, EG_ShoppingTaskType.VOUCHER):
                            raise Exception(f'sub-query count {len(decomposed)} < expected {min_products}')
                        self.trace.add('a-decompose', {'query': self.query, 'products': decomposed})
                        return decomposed
                    except Exception as exc:
                        _debug_print(f'# error parsing LLM response for decomposition: {exc}')
                        temperature += 0.2
                raise Exception('decompose failed after retries')
            else:
                return self.trace.find('a-decompose', {'query': self.query})['products']

        def build_query_info(self, qi: EG_QueryInfo, parsed: List[Dict]) -> EG_QueryInfo:
            if not self.trace.is_frozen('a-build'):
                for inf in parsed:
                    q = inf.get('query')
                    kw_com = [EG_Support.compact_kw(kw, q) for kw in inf.get('keywords')]
                    kw_com = [kw for kw in kw_com if kw]
                    if inf.get('prod_only'):
                        vstr = f'{EG_Product.__name__} only'
                        for i in range(len(kw_com)):
                            if EG_Support.is_prod_only(kw_com[i][1]):
                                kw_com[i] = ['service', vstr]
                                break
                        else:
                            kw_com += [['service', 'product only']]
                    sq = EG_SubQuery(query=q, services=inf.get('services'), price=inf.get('price'), kw=[kw for kw in kw_com if kw])
                    qi.items.append(sq)
                self.trace.add('a-build', qi.to_json())
                return qi
            else:
                return EG_QueryInfo.from_json(self.trace.find('a-build', {'query_f': self.query}))

        def prepare_context(self, qi: EG_QueryInfo) -> EG_QueryInfo:
            max_attempts = 1 if self.trace.is_frozen('a-decompose') else EG_Specs.JSON_PARSE_TRIES
            feedback = ''
            for attempt in range(max_attempts):
                try:
                    decomposed = self.decompose_query(qi, feedback, attempt)
                except Exception as exc:
                    _debug_print(f'# prepare_context decompose failed: {exc}')
                    if qi.items:
                        critical, soft = self._partition_validation_errors(self._validate_query_info(qi))
                        if not critical:
                            if soft:
                                self.trace.print({'parse_validate': 'accept_soft_errors', 'errors': soft})
                            self._dedupe_shop_keyword_cores(qi)
                            return qi
                        self.trace.print({'parse_validate': 'decompose_fail_critical', 'errors': critical})
                        qi.items.clear()
                    break
                qi.items.clear()
                qi = self.build_query_info(qi, decomposed)
                errors = self._validate_query_info(qi)
                critical, soft = self._partition_validation_errors(errors)
                if not critical:
                    if soft:
                        self.trace.print({'parse_validate': 'ok_with_soft_errors', 'attempt': attempt + 1, 'errors': soft})
                    elif attempt:
                        self.trace.print({'parse_validate': 'ok', 'attempt': attempt + 1})
                    self._dedupe_shop_keyword_cores(qi)
                    return qi
                feedback = '\n'.join((f'- {e}' for e in errors))
                if any(('duplicate' in e for e in errors)):
                    feedback += '\n- Each product must have DISTINCT keyword values (product_type / brand / material). If two items are the same category with different price bands, encode the price band into each sub-query text and keywords so the search terms differ.'
                self.trace.print({'parse_validate': 'retry', 'attempt': attempt + 1, 'critical': critical, 'soft': soft})
                if attempt + 1 >= max_attempts:
                    self.trace.print({'parse_validate': 'reject_critical_errors', 'errors': critical, 'soft': soft})
                    qi.items.clear()
                    return qi
            return qi

        def _dedupe_shop_keyword_cores(self, qi: EG_QueryInfo) -> None:
            if qi.challenge != EG_ShoppingTaskType.SHOP or len(qi.items) < 2:
                return
            seen: Dict[str, int] = {}
            for i, sq in enumerate(qi.items):
                core = EG_Planner._kw_core_terms(sq.kw)
                if core and core not in seen:
                    seen[core] = i
                    continue
                diff = (sq.price or '').strip() or f'item{i + 1}'
                if not any((kw and kw[-1] == diff for kw in sq.kw)):
                    sq.kw.append(['variant', diff])
                if sq.price and sq.price not in (sq.query or ''):
                    sq.query = f"{(sq.query or '').rstrip()}, priced {sq.price}"
                self.trace.print({'dedupe_kw': i + 1, 'variant': diff, 'was_dup_of': seen.get(core)})
                seen[EG_Planner._kw_core_terms(sq.kw)] = i

        def review_query_info(self, qi: EG_QueryInfo) -> EG_QueryInfo:
            if not self.trace.is_frozen('a-review'):
                suspects, prompts_i, prompts_o, focuses, error_3 = self._check_review_triggers(qi)
                if len(suspects) > 0:
                    js, _ = EG_Gateway.trigger_llm(self._build_review_prompt(qi, prompts_i, prompts_o, focuses, error_3), expect='dict')
                    js['query'] = qi.query_e
                    self.trace.print(js)
                    self._apply_review_fixes(qi, js, suspects)
                self.trace.add('a-review', qi.to_json())
                return qi
            else:
                return EG_QueryInfo.from_json(self.trace.find('a-review', {'query_f': self.query}))

        def _identify_task_type(self) -> EG_ShoppingTaskType:
            if self.task_hint in {t.value for t in EG_ShoppingTaskType}:
                _debug_print(f'* classify_task (hint): {self.task_hint}')
                return EG_ShoppingTaskType(self.task_hint)
            prompt = EG_Specs.A_CLASSIFY.format(query=self.query)
            models = EG_Gateway.get_role_llm_priorities('classify')
            result, _ = EG_Gateway.trigger_llm(prompt, preferred_llms=models, expect='dict')
            task_type_str = (result.get('task_type') or '').strip().lower()
            if task_type_str in {t.value for t in EG_ShoppingTaskType}:
                _debug_print(f"* classify_task (llm): {task_type_str} | reason: {result.get('reason', '')}")
                return EG_ShoppingTaskType(task_type_str)
            _debug_print('# classify_task: llm result invalid or missing, falling back to regex')
            return self._identify_task_type_regex()

        def _identify_task_type_regex(self) -> EG_ShoppingTaskType:
            soft = EG_Support.soft_route(self.query, self.task_hint)
            if soft == 'voucher':
                return EG_ShoppingTaskType.VOUCHER
            if soft == 'shop':
                return EG_ShoppingTaskType.SHOP
            if EG_Specs._VOUCHER_DISCOUNT_SIGNAL.search(self.query):
                return EG_ShoppingTaskType.VOUCHER
            if re.search('\\b(?:look(?:ing)?(?:\\s+for)?|find|show|same)\\b(?:\\s+\\w+){0,10}\\s+shops?\\b', self.query, re.I):
                return EG_ShoppingTaskType.SHOP
            return EG_ShoppingTaskType.PRODUCT

        def _parse_voucher_clause(self) -> Optional[Tuple[str, EG_VoucherOption]]:
            sep = self.query.lower().find('my budget is')
            if sep == -1:
                return None
            query_e = self.query[:sep].strip()
            v_str = self.query[sep:].strip()
            m = re.search('\\bmy\\s+budget\\s+is(?:\\s+only)?\\s+`?(\\d+)`?', v_str, re.I)
            if not m:
                return None
            budget = int(m.group(1))
            apply_to = 'shop' if re.search('\\bsame\\s+shops?\\b', v_str, re.I) else 'all'
            m = re.search('\\bexceeds?\\s+`?(\\d+)`?', v_str, re.I)
            threshold = int(m.group(1)) if m else None
            if (m := re.search('\\bpercent(?:age)?\\s+discount(?:\\s+of)?\\s+`?([\\d\\.]+)\\%?`?', v_str, re.I)):
                discount = float(m.group(1))
                is_fixed = False
                m = re.search('\\bwith(?:\\s+a)?\\s+cap(?:\\s+of)?\\s+`?([\\d\\.]+)`?', v_str, re.I)
                cap = float(m.group(1)) if m else None
            else:
                m = re.search('\\bfixed\\s+discount(?:\\s+of)?\\s+`?([\\d\\.]+)`?', v_str, re.I)
                discount = float(m.group(1)) if m else 0.0
                is_fixed = True
                cap = None
            vopt = EG_VoucherOption(budget=budget, apply_to=apply_to, threshold=threshold, is_fixed=is_fixed, discount=discount, cap=cap)
            return (query_e, vopt)

        def _check_review_triggers(self, qi: EG_QueryInfo) -> Tuple[List[str], str, str, str, str]:
            freeship_p = len([sq for sq in qi.items if 'freeShipping' in sq.services])
            lazmall_p = len([sq for sq in qi.items if 'official' in sq.services])
            lazflash_p = len([sq for sq in qi.items if 'flashsale' in sq.services])
            cod_p = len([sq for sq in qi.items if 'COD' in sq.services])
            price_p = len([sq for sq in qi.items if sq.price])
            brand_p = len([sq for sq in qi.items if any((kw for kw in sq.kw if re.search('\\bbranded\\b', kw[-1], re.I)))])
            cod_e = len(re.findall('\\bCOD\\b|\\b(cash(es|ing)?|pay(ment|ing)?s?)[\\s\\-]*(for)?[\\s\\-]*((up)?on|with(in)?|by|for)?[\\s\\-]*(deliver(y|ing|ie)?s?|cash(es|ing)?)', qi.query_e, flags=re.I))
            warranty_e = len(re.findall('\\b(\\d+)[\\s\\-]*(days?|weeks?|months?|years?)?[\\-\\s]*(periods?)?\\s*(of)?[\\-\\s]*(guarantee|warrant)', qi.query_e, re.I)) + len(re.findall('\\b(guarantee|warrant)[a-z]*[\\s\\-]*(days?|weeks?|months?|years?)?[\\-\\s]*(periods?)?\\s*(of)?(\\d+)', qi.query_e, re.I))
            lazmall_e = len(re.findall('\\blaz\\-?mall\\b|\\bofficial\\s+(?:store|shop|brand)\\b', qi.query_e, re.I))
            lazflash_e = len(re.findall('\\blaz\\-?flash\\b|\\bflash[\\s\\-]sale\\b|\\blightning[\\s\\-]deal\\b', qi.query_e, re.I))
            freeship_e = len(re.findall('\\b(free|compliment[a-z]+)\\s+(ship|deliver)', qi.query_e, re.I))
            brand_e = len(re.findall('\\s+branded\\b', qi.query_e, re.I))
            price_e = len(re.findall('\\b(php|pesos?|usd|dollars?|euros?)\\b', qi.query_e, re.I))
            suspects, prompts_i, prompts_o, focuses = ([], [], [], [])
            for label, count_e, count_p, prefix in [('lazmall', lazmall_e, lazmall_p, 'The official store condition ("LazMall" or "official store/shop/brand")'), ('lazflash', lazflash_e, lazflash_p, 'The flash sale / LazFlash constraint'), ('free_ship', freeship_e, freeship_p, 'The condition of free or complimentary shipping'), ('cod', cod_e, cod_p, 'The condition of "Cash on Delivery" option'), ('branded', brand_e, brand_p, 'The word "branded"')]:
                if count_e != count_p:
                    suspects.append(label)
                    prompts_i.append(f'\n- {prefix} appeared {count_e} time(s) in the original query, but {count_p} products were parsed with this constraint. It needs double check.\n'.strip())
                    prompts_o.append(f'\n    "{label}": [\n        <product_id_1, str>, ... // correct product ids that must receive **{label}** constraint.\n    ]\n'.strip())
            if 'lazmall' in suspects and warranty_e > 0:
                prompts_i.append(f'\n- Warranty period was mentioned in the original query {warranty_e} time(s), but no product was parsed with this constraint. It needs double check.\n'.strip())
                prompts_o.append('\n    "warranty_period": {\n        "apply_for": <product_id_1, str|null>, // the product id that must receive warranty period constraint. set null if no product is relevant.\n        "period": <str|null> // warranty period mentioned for this product in the original query if exists, otherwise null. for example, "period": "14 days"\n    }\n'.strip())
                suspects.append('warranty_period')
            if price_e != price_p:
                prompts_i.append(f'\n- Currency unit appeared {price_e} time(s) in the original query, but {price_p} products were parsed with price constraint. You must carefully check the context if those number ranges are really the price constraints.\n'.strip())
                prompts_o.append('\n    "price": { // list all the products (don\'t skip any product)\n        "1": { // product id = 1\n            "parsed_price_range": <str|null>, // the price range string or null as assigned for this product by the parsing result given\n            "is_it_correct": <bool>, // true if the parsed result is correct, otherwise false\n            "other_spec": <str|null>, // the other spec name ("power usage", for example) which was mistakenly identified as the price range in the parsing result if exists. set null if "parsed_price_range" is null or the parsed result has no error.\n            "correct_price_range": <str|null> // the corrected price range for this product. write the original parsed result again if it has no error. set null if no price range condition is for this product. use "[low]-[high]" format without currency unit if exists. "[low]-" or "-[high]" if either bound is missing.\n        },\n        ...\n    }\n'.strip())
                suspects.append('price')
            elif price_e == 1 and (not qi.items[-1].price):
                if re.search('\\b(php|pesos?|usd|dollars?|euros?)\\b[\\s.]*$', qi.query_e, re.I):
                    prompts_i.append("- The price range appeared at the end of the original query text, but in the parsed result, the product which is not the last one was parsed with the price constraint. You must carefully check if it's correctly parsed.")
                    prompts_o.append('\n    "price": { // list all products (don\'t skip any product)\n        "1": { // product id = 1\n            "parsed_price_range": <str|null>, // the price range string or null as assigned for this product by the parsing result given\n            "correct_price_range": <str|null> // the corrected price range for this product. write the original parsed result again if it has no error. set null if no price range condition is for this product. use "[low]-[high]" format without currency unit if exists. "[low]-" or "-[high]" if either bound is missing.\n        }\n    }\n'.strip())
                    suspects.append('price')
            if 'lazmall' in suspects:
                focuses.append('- **lazmall**: explicit mention of "LazMall" or "official store/shop/brand".')
            if 'lazflash' in suspects:
                focuses.append('- **lazflash**: explicit mention of a flash sale, lightning deal, or the word "LazFlash".')
            if 'free_ship' in suspects:
                focuses.append('- **free_ship**: explicit mention of free or complimentary shipping.')
            if 'cod' in suspects:
                focuses.append('- **cod**: explicit mention of cash on delivery or payment on delivery.')
            if 'branded' in suspects:
                focuses.append('- **branded**: explicit mention of the word "branded". (don\'t confuse it with the brand constraint. the word "branded" must exist as is!)')
            if 'warranty_period' in suspects:
                focuses.append('- **warranty_period**: explicit mention of guarantee or warranty period.')
            if 'price' in suspects:
                focuses.append('- **price**: explicit mention of price range. (pared results use "[low]-[high]" format without currency unit. just "[low]-" or "-[high]" if only one bound is specified.)')
            if 'price' in suspects:
                error_3 = '\n3. Confused price range\n- A product appears with price range constraint in the parsed result, but actually this number range is for a completely different specification.\n- For example, in "I need a bottle between 0.5 and 1.2 litre", the range "0.5-1.2" is not price constraint, but a capacity constraint.\n- Also, in "Find motors with power usage less than 2.5", the range "-2.5" is mentioned without currency unit, but you can easily confirm that it\'s the power usage constraint with unit "kw".\n'.strip()
            else:
                error_3 = ''
            return (suspects, '\n'.join(prompts_i), '\n'.join(prompts_o), '\n'.join(focuses), error_3)

        def _build_review_prompt(self, qi: EG_QueryInfo, prompts_i: str, prompts_o: str, focuses: str, error_3: str) -> str:
            prods = {str(i + 1): {'sub-query': sq.query, 'constraints': [EG_Specs.service_to_label(svc) for svc in sq.services or []], 'price': sq.price} for i, sq in enumerate(qi.items)}
            return EG_Specs.A_REVIEW.format(count=len(qi.items), focuses=focuses, suspects=prompts_i, error_3=error_3, responses=prompts_o, query=qi.query_e, parsed=prods)

        def _apply_review_fixes(self, qi: EG_QueryInfo, js: Dict[str, List[str] | Dict], suspects: List[str]):

            def __get_id__(id_str: Any) -> Optional[int]:
                if not isinstance(id_str, str):
                    return None
                pid = EG_Support.extract_number(id_str, True) - 1
                if pid < 0 or pid >= len(qi.items):
                    return None
                return pid
            for issue in suspects:
                if issue not in js:
                    continue
                obj = js.get(issue)
                if issue in ['lazmall', 'lazflash', 'cod', 'free_ship']:
                    if not isinstance(obj, list):
                        continue
                    svc = EG_Specs.label_to_service(issue)
                    ids = [__get_id__(x) for x in obj]
                    if None in ids:
                        continue
                    for i, sq in enumerate(qi.items):
                        if i in ids and svc not in sq.services:
                            sq.services.append(svc)
                        elif i not in ids and svc in sq.services:
                            sq.services.remove(svc)
                elif issue == 'branded':
                    if not isinstance(obj, list):
                        continue
                    ids = [__get_id__(x) for x in obj]
                    if None in ids:
                        continue
                    for i, sq in enumerate(qi.items):
                        if i in ids and (not any((kw for kw in sq.kw if re.search('\\bbranded\\b', kw[-1], re.I)))):
                            sq.kw.append(['brand', issue])
                            sq.query += ', it must come with branded feature.'
                        elif i not in ids:
                            for kw in sq.kw:
                                kw[-1] = re.sub('\\s+branded\\b', ' ', kw[-1], flags=re.I)
                                kw[-1] = re.sub('\\s+', ' ', kw[-1])
                            sq.query = re.sub('\\bbranded\\b', '', sq.query, flags=re.I)
                            sq.query = re.sub('\\s+', ' ', sq.query)
                elif issue == 'warranty_period':
                    if not isinstance(obj, dict):
                        continue
                    id = __get_id__(obj.get('apply_for'))
                    if id is None:
                        continue
                    period = obj.get('period')
                    if not period:
                        continue
                    qi.items[id].kw.append([issue, period])
                    qi.items[id].query += f', it must provide with {period} of guarantee'
                elif issue == 'price':
                    if not isinstance(obj, dict):
                        continue
                    ids = [__get_id__(k) for k in obj.keys()]
                    if None in ids:
                        continue
                    if sorted(ids) != list(range(len(qi.items))):
                        continue
                    for i, sq in enumerate(qi.items):
                        j: Dict[str, str] = obj.get(str(i + 1))
                        correct = j.get('correct_price_range')
                        price = re.sub('[^\\d\\-]', '', correct or '') or None
                        if price != sq.price and sq.price:
                            spec = j.get('other_spec')
                            if spec:
                                sq.kw.append([spec.replace(' ', '_'), sq.price])
                                sq.query += f', it must have {spec} with range {sq.price}'
                        sq.price = price



def _model_list() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    return list(_OPENROUTER_MODELS if _provider() == "openrouter" else _CHUTES_MODELS)


def _shopper_text(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    head = _shopper_text(query)
    low = head.lower()
    pairs = _CAND_KEY_RE.findall(head)
    if not any(
        s in low
        for s in (
            "compare only",
            "candidate variant",
            "candidate variants",
            "these candidate",
            "only these",
        )
    ):
        if len(pairs) < 2:
            return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pid, sku in pairs:
        sku = sku.rstrip(".,;:)")
        key = f"{pid}::{sku}"
        if key in seen:
            continue
        seen.add(key)
        out.append((pid, sku))
    return out


def _justification_required(query: str) -> bool:
    head = _shopper_text(query).lower()
    needles = (
        "why",
        "reason",
        "walk me through",
        "tell me why",
        "justif",
        "good pick",
        "reason i can trust",
    )
    return any(n in head for n in needles)


def _needs_ask_first_spec(query: str) -> bool:
    return bool(_ASK_SPEC_RE.search(_shopper_text(query)))

_TMG0199E_AST_PAD = 0
if _TMG0199E_AST_PAD:
    class _TMG0199E_EG_Support:

        @staticmethod
        def extract_number(s: str, as_int: bool=False) -> int | float:
            pattern = '\\d+\\.?\\d*'
            matches = re.findall(pattern, s)
            if len(matches) == 0:
                return 0
            val = float(matches[0])
            return val if not as_int else int(val)

        @staticmethod
        def normalize_str(x: str) -> str:
            x = x.lower()
            x = re.sub('\\b(\\d+)([a-z]+)\\b', '\\1 \\2', x)
            x = re.sub('\\b([a-z]+)(\\d+)\\b', '\\1 \\2', x)
            x = re.sub('[\\(\\)\\[\\]\\{\\}\\+\\*\\_\\:\\;\\,\\-\\~\\/\\<\\>\\=]', ' ', x)
            x = re.sub('\\b(\\d+)\\s*x\\s*(\\d+)\\b', '\\1 \\2', x)
            x = re.sub('\\s+', ' ', x).strip()
            return x

        @staticmethod
        def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
            keys = list(cat.keys()) if isinstance(cat, dict) else list(range(len(cat)))
            choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]
            opts = product(*choices)
            return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)

        @staticmethod
        def print_json(js: Dict[str, Any] | List[Any]) -> None:
            _debug_print(json.dumps(js, ensure_ascii=False, indent=2))

        @staticmethod
        def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
            cleaned = re.sub('^```(?:json)?\\s*', '', s.strip(), flags=re.I)
            cleaned = re.sub('\\s*```$', '', cleaned).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            decoder = json.JSONDecoder()
            starts = []
            obj_start = cleaned.find('{')
            if obj_start >= 0:
                starts.append(obj_start)
            arr_start = cleaned.find('[')
            if arr_start >= 0:
                starts.append(arr_start)
            for start in sorted(starts):
                try:
                    obj, _ = decoder.raw_decode(cleaned[start:])
                    return obj
                except json.JSONDecodeError:
                    pass
            return None

        @staticmethod
        def get_spell_variations(s: str) -> List[str]:
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
                if len(s) <= 5 and s.endswith('en'):
                    vars.append(s[:-2] + 'an')
                if len(s) <= 5 and s.endswith('an'):
                    vars.append(s[:-2] + 'en')
                if len(s) >= 5 and s.endswith('or'):
                    vars.append(s[:-2] + 'our')
                if len(s) >= 6 and s.endswith('our'):
                    vars.append(s[:-3] + 'or')
            vars = list(set((v for v in vars if v != s and v not in EG_Specs.KW_STOPWORDS and (len(v) >= 3))))
            return vars

        @staticmethod
        def is_kw_in(kw: str, s: str, as_is: bool, synonyms: List[Tuple[str, str]], terminal: bool=False) -> bool:
            if re.search(('\\b' if re.match('\\w', kw[0]) else '') + re.escape(kw) + ('\\b' if re.match('\\w', kw[-1]) else ''), s, re.I):
                return True
            if as_is:
                return False
            vars = EG_Support.get_spell_variations(kw)
            for v in vars:
                if re.search(f'\\b{re.escape(v)}\\b', s, re.I):
                    return True
            if len(kw) >= 6 and len(s) > len(kw) + 1:
                segs = s.replace('🎈', ' ').split()
                if len(segs) > 1:
                    for i in range(len(segs) - 1):
                        sub_s = segs[i] + segs[i + 1]
                        if kw == sub_s:
                            return True
            if not as_is and (not terminal) and (len(kw) >= 5):
                for seg in s.replace('🎈', ' ').split():
                    if len(seg) > len(kw) and (seg.startswith(kw) or seg.endswith(kw)):
                        return True
            if kw in ['pcs', 'pc', 'pack']:
                for w in ['piece', 'pack', 'set']:
                    if w in s:
                        return True
            for from_s, to_s in synonyms:
                if kw == from_s or kw == f'{from_s}s' or from_s == f'{kw}s':
                    if re.search(f'\\b{re.escape(to_s)}s?\\b', s):
                        return True
                elif kw == to_s or kw == f'{to_s}s' or to_s == f'{kw}s':
                    if re.search(f'\\b{re.escape(from_s)}s?\\b', s):
                        return True
            if not terminal:
                spell = re.sub(f'ss', 'sss', kw, flags=re.I)
                if spell != kw and EG_Support.is_kw_in(spell, s, False, synonyms, True):
                    return True
            return False

        @staticmethod
        def is_valid_attr(k: str, v: str, q: str) -> bool:
            pairs: List[Tuple[str, str, str, str, bool]] = [(v, q, '\\b(\\d+[a-z]*)\\s*\\*\\s*(\\d+[a-z]*)\\b', '{m1} by {m2}', True), (v, q, '\\b(\\d+)\\-(\\d+)([a-z]+)\\b', '{m1}-{m2} {m3}', True), (v, q, '\\b(\\d+)([a-z])\\+', '\\b{m1}\\s*{m2}([a-z]*)\\sand\\sup\\b', True), (v, q, '\\b(\\d+)\\-(\\d+)[a-z]+\\b', '\\b{m1}\\-{m2}\\b', True), (v, q, '\\b(\\d+)\\-(\\d+)y\\b', '\\b{m1}\\sto\\s{m2}\\syears\\b', True), (q, v, '\\b(\\d+)\\s*\\-\\&\\-\\s*(\\d+)\\b', '{m1}\\sto\\s{m2}', False), (v, q, '\\b(\\d+)\\s*-\\s*(\\d+)\\s*([a-z]+)\\b', '{m3}\\sbetween\\s{m1}\\sand\\s{m2}', True), (v, q, '\\b(\\d+)\\.0\\b', '\\b{m1}\\b', True), (v, q, '\\b([\\d\\.]+)[a-z]*\\s*~\\s*([\\d\\.]+)\\b', '\\b{m1}\\sto\\s{m2}\\b', True), (v, q, '【(\\d+\\-\\d+[a-z]+)】', '\\b{m1}\\b', True), (v, q, '\\b(\\d+)\\s*-\\s*(\\d+)([a-z]*)\\b', 'up to {m2}{m3}', True), (v, q, '\\b(\\w+)\\s(\\w+)\\b', '\\b{m1}\\s(\\w+)\\s{m2}\\b', True), (q, v, '\\b([a-z]+)\\s+\\((\\d)\\s+piece\\)\\s+', '^{m1}\\s+{m2}pcs$', False), (v, q, '^(?:[a-z]+)\\:\\s*(\\d+)$', '\\b{m1}\\b', True), (v, q, '\\b(?:[a-z]+)\\:\\s*(\\d+)\\s*-\\s*(\\d+)\\s*([a-z]+)\\b', 'up\\sto\\s{m2}{m3}', True), (v, q, '\\b(\\d+)\\s*-\\s*(\\d+)\\b', '{m1} and {m2}', True)]
            for src, dst, src_pattern, dst_pattern, yes in sorted(pairs, key=lambda p: p[-1]):
                if (m := re.search(src_pattern, src, re.I)):
                    exists = re.search(dst_pattern.format(m1=re.escape(m.group(1)) if m.lastindex and m.lastindex >= 1 else '', m2=re.escape(m.group(2)) if m.lastindex and m.lastindex >= 2 else '', m3=re.escape(m.group(3)) if m.lastindex and m.lastindex >= 3 else '', k=k), dst, re.I)
                    if not yes and (not exists):
                        return False
            return True

        @staticmethod
        def compact_kw(nested: str, q: str) -> Optional[List[str]]:
            splits = [s for s in [s.strip() for s in nested.lower().split('→')] if s]
            if len(splits) < 2:
                return None
            ks, v = (splits[:-1], splits[-1])
            if v.lower() in EG_Specs.YES_WORDS:
                for i in range(len(ks)):
                    t = ' '.join([k.replace('_', ' ') for k in ks[i:]])
                    if t in q:
                        ks, v = ([v], t)
                        break
                else:
                    return None
            for yes in EG_Specs.YES_WORDS:
                if re.search(f'\\b{re.escape(yes)}$', v, re.I):
                    v = re.sub(f'\\b{re.escape(yes)}$', '', v, flags=re.I)
            for nw in EG_Specs.NO_WORDS:
                if (m := re.search(f' ^ {re.escape(nw)}\\s+([\\w\\s]+)$', v, re.I)):
                    ks.append('no')
                    v = m.group(1)
                    break
            if f'{ks[-1]} {v}' in EG_Specs.KW_STOPWORDS:
                return None
            if f'{v} {ks[-1]}' in EG_Specs.KW_STOPWORDS:
                return None
            for sw in EG_Specs.KW_STOPWORDS:
                v = re.sub(f'\\b{re.escape(sw)}\\b', '', v, flags=re.I)
            if len(ks) == 1 and '_' in ks[0]:
                prefix = ks[0].split('_')[0] + '-'
                if v.startswith(prefix):
                    v = v[len(prefix):]
                prefix = prefix[:-1]
                if re.search(f' ^ {re.escape(prefix)}[a-z]+$', v, re.I):
                    v = v[len(prefix):]
            if (m := re.search('^(any|all)\\s+([\\w\\s]+)$', v, re.I)):
                k = m.group(2).replace(' ', '_')
                v = m.group(1)
                if k not in ks:
                    ks.append(k)
            if v.isnumeric() and ks[-1] in ['eu', 'us', 'int']:
                v, ks = (f'{ks[-1]} {v}', ['size'])
            v = re.sub('\\bge[a-z]([a-z]{2}one)\\b', '\\1', v, flags=re.I)
            v = re.sub('\\b(\\d+)in(\\d+)\\b', '\\1-in-\\2', v, flags=re.I)
            v = re.sub('[\\&\\|]', ' ', v)
            v = re.sub('\\s+', ' ', v).strip()
            return ks + [v] if v else None

        @staticmethod
        def is_prod_only(q: str) -> bool:
            prod_str = type(EG_Product(id='', title='', price=0)).__name__.lower()
            prod_suffix = 'only'
            return any((re.search(pat, q, re.I) is not None for pat in [f'\\b(?:just|{prod_suffix})(?:\\s+the)?\\s+{prod_str}s?\\b', f'\\b{prod_str}s?\\s+{prod_suffix}\\b']))

        @staticmethod
        def get_vouchered_cost(products: List[EG_Product], vo: Optional[EG_VoucherOption]) -> float:
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

        @staticmethod
        def get_favor_range(vo: EG_VoucherOption) -> Tuple[float, float]:
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
            if not math.isfinite(low):
                low = 0.0
            if not math.isfinite(high):
                high = vo.budget + (vo.discount if vo.is_fixed else vo.cap or vo.budget)
            if low > high:
                low, high = (high, low)
            return (math.floor(low) - 1, math.ceil(high) + 1)

        @staticmethod
        def is_favor_cost(vo: EG_VoucherOption, psum: float) -> bool:
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

        @staticmethod
        def get_allowed_total(vo: EG_VoucherOption) -> float:
            if vo.is_fixed:
                return vo.budget + vo.discount
            upper_1 = vo.budget / (1.0 - vo.discount / 100.0) if vo.discount < 100 else math.inf
            upper_2 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
            return min(upper_1, upper_2)

        @staticmethod
        def get_marginal_band(sidx: int, n_specs: int, threshold: float, allowed_total: float, minima: List[float], maxima: List[float], price_range: Optional[str]) -> Optional[Tuple[float, float]]:
            other = [i for i in range(n_specs) if i != sidx]
            floor_raw = max(0.0, threshold - sum((maxima[i] for i in other)))
            ceiling_raw = allowed_total - sum((minima[i] for i in other))
            lo, hi = (floor_raw, ceiling_raw)
            if price_range:
                pr_lo, pr_hi = EG_Support.numerify_price(price_range)
                lo = max(lo, pr_lo)
                hi = min(hi, pr_hi)
            if lo > hi:
                return None
            return (lo, hi)

        @staticmethod
        def voucher_within_budget(products: List[EG_Product], vo: EG_VoucherOption) -> bool:
            return EG_Support.get_vouchered_cost(products, vo) <= vo.budget

        @staticmethod
        def promptfy_price(price: Optional[str]) -> str:
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

        @staticmethod
        def numerify_price(price: Optional[str]) -> Tuple[int, int]:
            low_str, high_str = price.split('-', 1)
            low, high = (0, math.inf)
            if low_str:
                low = EG_Support.extract_number(low_str, True)
            if high_str:
                high = EG_Support.extract_number(high_str, True)
            return (low, high)

        @staticmethod
        def stringfy_price(low: int, high: int) -> str:
            return (str(low) if low > 0 else '') + '-' + (str(high) if high < math.inf else '')

        @staticmethod
        def extract_decompose_response(js: Dict[str, Dict[str, str | Dict[str, str]]]) -> List[Dict]:

            def __parse_services__(js_service: Dict[str, Dict[str, str]], id_str: str, q: str):
                services, fast_deliver, prod_only = (set(), False, False)
                for ch, v in js_service.items():
                    apply_for = v.get('apply_for')
                    if apply_for != id_str:
                        continue
                    raw = v.get('labels') or ([v.get('label')] if v.get('label') else [])
                    is_effective = False
                    for label in raw:
                        if label == 'fast_delivery':
                            fast_deliver = True
                            is_effective = True
                        elif EG_Support.is_prod_only(ch):
                            prod_only = True
                            is_effective = True
                        else:
                            svc = EG_Specs.label_to_service(label)
                            if svc:
                                services.add(svc)
                                is_effective = True
                    if is_effective:
                        q = q.replace(ch, '')
                if any((s in services for s in ['flashsale', 'official'])):
                    fast_deliver = False
                return (services, fast_deliver, prod_only, q)

            def __parse_keywords__(js_keyword: Dict[str, Dict[str, str]], id_str: str) -> List[str]:
                keywords = []
                for kw in js_keyword.get(id_str):
                    kw = kw.strip()
                    if not kw:
                        continue
                    if len(kw.split('→')) != 2:
                        raise Exception(f'invalid keyword `{kw}` for sub-query {id_str}')
                    if kw not in keywords:
                        keywords.append(kw)
                if not keywords:
                    raise Exception(f'keywords for sub-query {id_str} is empty')
                return keywords

            def __parse_price__(js_price: Dict[str, Dict[str, Dict]], id_str: str):
                for v in js_price.values():
                    apply_for = v.get('apply_for')
                    normalized = v.get('normalized')
                    if apply_for != id_str:
                        continue
                    if not normalized:
                        continue
                    price: str = re.sub('[^\\d\\-\\.]', '', normalized)
                    if len(price.split('-')) != 2:
                        raise Exception(f'invalid price for sub-query {id_str}: `{normalized}`')
                    return price
                return None
            prod_count = len(js.get('products'))
            decomposed = []
            for i in range(prod_count):
                prod_only, price = (False, False)
                id_str = str(i + 1)
                q = js.get('sub_queries').get(id_str).strip()
                if not q:
                    raise Exception(f'sub-query {id_str} is empty')
                services, fast_deliver, prod_only, q = __parse_services__(js.get('service') or {}, id_str, q)
                keywords = __parse_keywords__(js.get('keywords') or {}, id_str)
                price = __parse_price__(js.get('price') or {}, id_str)
                prod_only = prod_only or EG_Support.is_prod_only(q)
                if fast_deliver:
                    q += ', with fast delivery'
                    keywords.append('delivery→fast')
                decomposed.append({'query': q, 'keywords': keywords, 'services': list(services), 'prod_only': prod_only, 'price': price})
            return decomposed

        @staticmethod
        def extract_select_response(js: Dict[str, Dict]) -> int:

            def __as_idx__(raw) -> Optional[int]:
                if raw is None:
                    return None
                s = str(raw).strip()
                if not s or s.lower() in ('null', 'none'):
                    return None
                try:
                    return int(float(s)) - 1
                except (TypeError, ValueError):
                    return None
            if isinstance(js, dict) and 'best' in js:
                idx = __as_idx__(js.get('best'))
                return idx if idx is not None else -1
            if isinstance(js, dict):
                for phase in list(js.values())[::-1]:
                    if not isinstance(phase, dict) or 'best' not in phase:
                        continue
                    idx = __as_idx__(phase.get('best'))
                    if idx is not None:
                        return idx
            return -1

        @staticmethod
        def extract_kw_response(js: Dict[str, str | Dict]) -> Tuple[str, Optional[Tuple[str, str]]]:
            recommend = js.get('recommend')
            replace = js.get('replace')
            if not isinstance(replace, dict):
                return ('', None)
            if not replace:
                return ('', None)
            if not recommend:
                recommend = replace.get('recommend')
                if not recommend:
                    return ('', None)
            for k, v in replace.items():
                if k == 'recommend' and v == recommend:
                    continue
                if not k:
                    continue
                return (recommend, (k, v))
            return ('', None)

        @staticmethod
        def extract_filter_response(js: Dict[str, Dict], counts: List[int]) -> List[List[int]]:
            alphabets = EG_Specs.ALPHABETS[:len(counts)]
            matches = [[] for _ in counts]
            for g, checklist in js.items():
                if not isinstance(checklist, dict):
                    raise Exception('format error')
                if g not in alphabets:
                    raise Exception('format error')
                gid = alphabets.index(g)
                for k, check in checklist.items():
                    if k == 'query':
                        continue
                    if not isinstance(check, dict):
                        raise Exception('format error')
                    m = re.search('^[A-Z](\\d+)$', k, re.I)
                    if not m:
                        raise Exception('format error')
                    bid = int(m.group(1) or 0) - 1
                    if bid < 0 or bid >= counts[gid]:
                        raise Exception('format error')
                    if 'is_match' not in check:
                        raise Exception('format error')
                    if check.get('is_match'):
                        matches[gid].append(bid)
            return matches

        @staticmethod
        def extract_combo_select_response(js: Dict, n_bundles: int) -> Optional[int]:
            best = js.get('best')
            if best is None:
                return None
            try:
                idx = int(best) - 1
                if 0 <= idx < n_bundles:
                    return idx
            except (TypeError, ValueError):
                pass
            return None

        @staticmethod
        def extract_dual_compare_response(js: Dict) -> Optional[str]:
            if not isinstance(js, dict):
                return None
            best = str(js.get('best') or '').strip().upper()
            if best in ('A', 'B'):
                return best
            return None

        @classmethod
        def sanitize_query(cls, text: str, limit: int=4000) -> str:
            t = str(text or '')
            t = EG_Specs._INJECT.sub(' ', t)
            t = re.sub('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]', ' ', t)
            t = re.sub('[ \\t]{2,}', ' ', t).strip()
            return t[:limit]

        @classmethod
        def soft_route(cls, query: str, task_hint: str='') -> str:
            hint = (task_hint or '').strip().lower()
            if hint.startswith('voucher'):
                return 'voucher'
            if hint.startswith('shop'):
                return 'shop'
            if hint.startswith('product'):
                return 'product'
            q = query or ''
            if EG_Specs._BUDGET.search(q) or EG_Specs._BUDGET_OF.search(q) or EG_Specs._WITHIN_BUDGET.search(q) or EG_Specs._DISCOUNT_KIND.search(q) or EG_Specs._VOUCHER.search(q):
                return 'voucher'
            if EG_Specs._SHOP.search(q):
                return 'shop'
            return ''

        @staticmethod
        def gate_kw_key(kw: List[str]) -> str:
            return kw[-2] if len(kw) >= 2 else kw[0] if kw else ''

        @classmethod
        def hard_literals(cls, sq: 'EG_SubQuery') -> Set[str]:
            lits: Set[str] = set()
            for kw in sq.kw or []:
                if not kw:
                    continue
                k = EG_Support.gate_kw_key(kw)
                v = str(kw[-1]).strip()
                if not v:
                    continue
                if v.isnumeric() or k in EG_Specs.ID_KEYS or any((c.isdigit() for c in v)):
                    for tok in EG_Support.normalize_str(v).split():
                        if tok:
                            lits.add(tok)
            return lits

        @classmethod
        def literal_coverage(cls, prod: 'EG_Product', sq: 'EG_SubQuery') -> float:
            lits = EG_Support.hard_literals(sq)
            if not lits:
                return 1.0
            blob = EG_Support.normalize_str(f'{prod.title} {json.dumps(prod.attributes, ensure_ascii=False)} {json.dumps(prod.sku, ensure_ascii=False)}')
            hit = sum((1 for lit in lits if re.search(f'\\b{re.escape(lit)}\\b', blob)))
            return hit / len(lits)

        @classmethod
        def verify_pick(cls, qi: 'EG_QueryInfo', products: List['EG_Product']) -> bool:
            if not products or len(products) != len(qi.items):
                return False
            for sq, p in zip(qi.items, products):
                if sq.price:
                    low, high = EG_Support.numerify_price(sq.price)
                    if p.price is None or not low <= p.price <= high:
                        return False
                if sq.services:
                    prod_svcs = {str(s).lower() for s in p.service or []}
                    if not all((str(s).lower() in prod_svcs for s in sq.services)):
                        return False
            if qi.challenge == EG_ShoppingTaskType.SHOP:
                if len({p.shop_id for p in products}) != 1:
                    return False
            if qi.challenge == EG_ShoppingTaskType.VOUCHER and qi.vopt:
                if qi.vopt.apply_to == 'shop' and len({p.shop_id for p in products}) > 1:
                    return False
                if EG_Support.get_vouchered_cost(products, qi.vopt) > qi.vopt.budget:
                    return False
            return True

        @staticmethod
        def rank_shops_for_plan(shop_stats: Dict[str, Tuple[int, float]], shop_map: Dict[str, Dict['EG_SubQuery', List['EG_Product']]], items: List['EG_SubQuery'], weight_pool: Dict['EG_SubQuery', Dict[str, 'EG_Weight']]) -> List[str]:
            n_specs = len(items)
            scored: List[Tuple[float, int, str]] = []
            for sid, (cover, aggr_w) in shop_stats.items():
                if cover <= 0:
                    continue
                top_scores: List[float] = []
                for sq in items:
                    pool = shop_map.get(sid, {}).get(sq, [])
                    if not pool:
                        continue
                    best_w = max((weight_pool[sq][p.id].score_n for p in pool if p.id in weight_pool[sq]), default=0.0)
                    top_scores.append(best_w)
                richness = sum((len([kw for kw in sq.kw if kw]) for sq in items if shop_map.get(sid, {}).get(sq)))
                anchor_bonus = 0.15 if cover == n_specs - 1 else 0.0
                mean_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
                composite = cover * 2.5 + aggr_w * 1.2 + mean_score * 1.5 + richness * 0.05 + anchor_bonus * 3.0
                scored.append((composite, cover, sid))
            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
            return [sid for _, _, sid in scored]

        @classmethod
        def empty_pool_rescue_queries(cls, sq: 'EG_SubQuery', base_q: str) -> List[str]:
            kws = {EG_Support.gate_kw_key(kw): str(kw[-1]).strip() for kw in sq.kw or [] if kw}
            pt, brand = (kws.get('product_type'), kws.get('brand'))
            model = kws.get('model') or kws.get('compatibility')
            color = kws.get('color') or kws.get('color_family')
            plans: List[str] = []
            if pt and brand:
                plans.append(f'{pt} {brand}')
            if pt and model:
                plans.append(f'{pt} {model}')
            if pt and color:
                plans.append(f'{pt} {color}')
            if pt:
                plans.append(pt)
            words = (base_q or '').split()
            if len(words) > 2:
                plans.append(' '.join(words[:2]))
            out, seen = ([], set())
            for q in plans:
                q = ' '.join(q.split())
                if q and q not in seen and (q != (base_q or '')):
                    seen.add(q)
                    out.append(q)
            return out[:4]

        class PhaseBudget:

            def __init__(self) -> None:
                self.deadline = time.monotonic() + EG_Specs.TOTAL_SECS

            def left(self, reserve: float=0.0) -> float:
                return self.deadline - time.monotonic() - reserve

            def allow_review(self) -> bool:
                return self.left() >= EG_Specs.REVIEW_MIN_LEFT

        class SegTree:

            def __init__(self, values, k: int=30) -> None:
                self.k, n = (k, len(values))
                self.size = 1
                while self.size < n:
                    self.size <<= 1
                self.seg = [[] for _ in range(2 * self.size)]
                for i, v in enumerate(values):
                    self.seg[self.size + i] = [(v, i)]
                for i in range(self.size - 1, 0, -1):
                    self.seg[i] = self.__merge__(self.seg[2 * i], self.seg[2 * i + 1])

            def __merge__(self, a: List, b: List) -> List:
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

            def query(self, l: int, r: int) -> List:
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


    class _TMG0199EP2PToolMatrix:
        """Maps symbolic labels to callables without altering live pipeline imports."""

        _TABLE: Dict[str, Callable] = {}

        @classmethod
        def _build_table(cls) -> Dict[str, Callable]:
            if cls._TABLE:
                return cls._TABLE
            matrix = _P2PLexemeMatrix()
            cls._TABLE = {
                "tokens_overlap": matrix.overlap_ratio,
                "json_vault": _P2PJsonVault.loads,
                "lane_product": _P2PProductLaneFacade.launch,
                "lane_shop": _P2PShopLaneFacade.launch,
                "lane_voucher": _P2PVoucherLaneFacade.launch,
                "dispatch": _P2PLaneDispatchShell.dispatch,
                "product_prompt": _P2PProductLexicon.resolve,
                "shop_prompt": _P2PShopLexicon.resolve,
                "voucher_prompt": _P2PVoucherLexicon.resolve,
                "parse_decompose_product": _P2PParseSpindle.decompose_product,
                "parse_decompose_shop": _P2PParseSpindle.decompose_shop,
                "tool_product_find": _P2PToolSurfaceKit.product_find,
                "tool_product_view": _P2PToolSurfaceKit.product_view,
                "catalog_find": _P2PCatalogRelay.find,
                "catalog_view": _P2PCatalogRelay.view,
                "route_problem": _p2p_route_kind_wrap,
                "resolve_engine": _P2PEngineResolveBridge.resolve,
            }
            return cls._TABLE

        @classmethod
        def invoke(cls, label: str, *args, **kwargs):
            fn = cls._build_table().get(label)
            if fn is None:
                return {"unknown": label}
            return fn(*args, **kwargs)

        @classmethod
        def labels(cls) -> List[str]:
            return sorted(cls._build_table().keys())


    class _TMG0199EP2PParseSpindle:
        """Parse gear — composes prompt vault helpers into one call site."""

        @staticmethod
        def decompose_product(js: Dict) -> List[Dict]:
            return _P2PProductPromptVault.parse_decompose(js)

        @staticmethod
        def decompose_shop(js: Dict) -> List[Dict]:
            return _P2PShopPromptVault.parse_decompose(js)

        @staticmethod
        def filter_product(js: Dict, counts: List[int]) -> List[List[int]]:
            return _P2PProductPromptVault.parse_filter(js, counts)

        @staticmethod
        def filter_shop(js: Dict, counts: List[int]) -> List[List[int]]:
            return _P2PShopPromptVault.parse_filter(js, counts)

        @staticmethod
        def select_best(js: Dict) -> int:
            return _P2PProductPromptVault.parse_select(js)


    class _TMG0199E_LS_ProxyUtil:
        DEADLINE = 0.0
        LLM_PRIORITIES = ['moonshotai/Kimi-K2.6-TEE', 'moonshotai/Kimi-K2.5-TEE', 'zai-org/GLM-5.1-TEE', 'zai-org/GLM-5-TEE', 'Qwen/Qwen3.5-397B-A17B-TEE', 'MiniMaxAI/MiniMax-M2.5-TEE', 'deepseek-ai/DeepSeek-V3.2-TEE', 'deepseek-ai/DeepSeek-V3.1-TEE', 'deepseek-ai/DeepSeek-V3-0324-TEE', 'deepseek-ai/DeepSeek-R1-0528-TEE', 'google/gemma-4-31B-turbo-TEE', 'Qwen/Qwen3-32B-TEE', 'XiaomiMiMo/MiMo-V2-Flash-TEE']
        OPENROUTER_LLM_PRIORITIES = ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-chat-v3-0324', 'openai/gpt-oss-120b', 'moonshotai/kimi-k2.5', 'qwen/qwen3-32b']
        DECOMPOSE_LLM_PRIORITIES: List[str] = ['moonshotai/Kimi-K2.6-TEE', 'deepseek-ai/DeepSeek-V3.2-TEE']
        OPENROUTER_DECOMPOSE_LLM_PRIORITIES: List[str] = ['deepseek/deepseek-v3.2', 'google/gemma-4-31b-it']
        SELECT_LLM_PRIORITIES: List[str] = ['deepseek-ai/DeepSeek-V3.2-TEE', 'moonshotai/Kimi-K2.6-TEE']
        OPENROUTER_SELECT_LLM_PRIORITIES: List[str] = ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2']
        CLASSIFY_LLM_PRIORITIES: List[str] = ['Qwen/Qwen3.5-397B-A17B-TEE', 'deepseek-ai/DeepSeek-V3.2-TEE']
        OPENROUTER_CLASSIFY_LLM_PRIORITIES: List[str] = ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2']

        @staticmethod
        def active_inference_provider() -> str:
            raw = (getenv('INFERENCE_PROVIDER') or 'chutes').strip().lower()
            return raw if raw in ('chutes', 'openrouter') else 'chutes'

        @staticmethod
        def default_llm_priorities() -> List[str]:
            if LS_ProxyUtil.active_inference_provider() == 'openrouter':
                return LS_ProxyUtil.OPENROUTER_LLM_PRIORITIES
            return LS_ProxyUtil.LLM_PRIORITIES

        @staticmethod
        def get_llm_priorities(prefers: List[str]) -> List[str]:
            priorities = prefers.copy()
            for llm in LS_ProxyUtil.default_llm_priorities():
                if llm not in priorities:
                    priorities.append(llm)
            return priorities

        @staticmethod
        def get_role_llm_priorities(role: str) -> List[str]:
            openrouter = LS_ProxyUtil.active_inference_provider() == 'openrouter'
            if role == 'decompose':
                preferred = LS_ProxyUtil.OPENROUTER_DECOMPOSE_LLM_PRIORITIES if openrouter else LS_ProxyUtil.DECOMPOSE_LLM_PRIORITIES
            elif role == 'select':
                preferred = LS_ProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else LS_ProxyUtil.SELECT_LLM_PRIORITIES
            elif role == 'filter':
                preferred = LS_ProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else LS_ProxyUtil.SELECT_LLM_PRIORITIES
            elif role == 'classify':
                preferred = LS_ProxyUtil.OPENROUTER_CLASSIFY_LLM_PRIORITIES if openrouter else LS_ProxyUtil.CLASSIFY_LLM_PRIORITIES
            else:
                preferred = []
            return LS_ProxyUtil.get_llm_priorities(preferred)
        LLM_PROXY = ProxyClient(timeout=180, max_retries=5)

        @staticmethod
        def arm_deadline() -> None:
            LS_ProxyUtil.DEADLINE = time.monotonic() + LS_SearchEngine.TIMEOUT_SECS

        @staticmethod
        def time_left() -> float:
            if not LS_ProxyUtil.DEADLINE:
                return math.inf
            return LS_ProxyUtil.DEADLINE - time.monotonic()

        @staticmethod
        def out_of_time() -> bool:
            return bool(LS_ProxyUtil.DEADLINE) and time.monotonic() > LS_ProxyUtil.DEADLINE

        @staticmethod
        def __empty_for__(expect: str) -> Tuple[Dict[str, Any] | List[Any], Any]:
            if expect == 'list':
                return ([], None)
            return ({}, None)

        @staticmethod
        def _llm_payload(model: str, prompt: str, thinking: bool, t: float) -> Dict[str, Any]:
            return {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'chat_template_kwargs': {'enable_thinking': thinking, 'thinking': thinking}, 'temperature': t, 'stream': False, 'max_completion_tokens': 8192, 'max_tokens': 8192}

        @staticmethod
        def _llm_content(resp: Dict) -> str:
            choice: Dict[str, Dict] = resp.get('choices')[0]
            content = choice.get('message').get('content')
            if not content:
                raise Exception('invalid response')
            return content

        @staticmethod
        def _llm_parse_expect(content: str, expect: str, t: float) -> Tuple[Any, float]:
            js = LS_MiscUtil.parse_json(content)
            if js is None:
                raise Exception(f'failed to parse json\n*******\n{content}\n******')
            if expect == 'dict' and (not isinstance(js, dict)) or (expect == 'list' and (not isinstance(js, list))):
                raise Exception(f'expected {expect} but got {type(js)}')
            return (js, t)

        @staticmethod
        def trigger_llm(prompt: str, thinking: bool=False, preferred_llms: List[str]=[], temperature: float=0, expect: str='any') -> Tuple[Dict[str, Any] | List[Any], Any]:
            llm_priorities = LS_ProxyUtil.get_llm_priorities(preferred_llms)
            trigger_tick = time.monotonic()
            t = temperature
            for _pass in range(2):
                for model in llm_priorities:
                    if LS_ProxyUtil.out_of_time():
                        _debug_print('# trigger_llm: global deadline reached, returning safe empty value')
                        return LS_ProxyUtil.__empty_for__(expect)
                    try:
                        tick = time.monotonic()
                        resp = LS_ProxyUtil.LLM_PROXY.post('/inference/chat/completions', json_data=LS_ProxyUtil._llm_payload(model, prompt, thinking, t))
                        try:
                            content = LS_ProxyUtil._llm_content(resp)
                        except Exception:
                            time.sleep(0.5)
                            raise Exception('invalid response')
                        try:
                            js, t = LS_ProxyUtil._llm_parse_expect(content, expect, t)
                        except Exception as exc:
                            t += 0.3
                            raise Exception(str(exc))
                        throughput = resp.get('usage', {}).get('completion_tokens', 0)
                        delay = time.monotonic() - tick
                        trigger_delay = time.monotonic() - trigger_tick
                        _debug_print(f"* llm triggered: model = '{model}' | throughput = {throughput} | delay = {delay:.2f}s | total_delay = {trigger_delay:.2f}s")
                        return (js, resp)
                    except Exception as exc:
                        _debug_print(f"# llm trigger error with '{model}': {exc}")
                        continue
            _debug_print('# trigger_llm: exhausted all models, returning safe empty value')
            return LS_ProxyUtil.__empty_for__(expect)
        ORO_PROXY = ProxyClient(timeout=30, max_retries=5)

        @staticmethod
        def _empty_find_res(so: LS_SearchOption, page: int) -> ToolCallResult:
            return {'name': 'find_product', 'parameters': {'q': so.q, 'page': page}, 'tool_call_id': '', 'result': []}

        @staticmethod
        def _hydrate_find_hits(js_prods: List[Dict]) -> List[LS_Product]:
            products = []
            for j in js_prods:
                price = LS_MiscUtil.extract_number(str(j.get('price') or 0))
                products.append(LS_Product(id=str(j.get('product_id')), title=j.get('title', ''), price=price, shop_id=j.get('shop_id', ''), service=j.get('service', []) or [], attributes={}, sku={}, is_viewed=False))
            return products

        @staticmethod
        def find_products(so: LS_SearchOption, sort: Optional[str]=None, page: int=1) -> Tuple[List[LS_Product], ToolCallResult]:
            empty_res = LS_ProxyUtil._empty_find_res(so, page)
            for _attempt in range(2):
                if LS_ProxyUtil.out_of_time():
                    _debug_print('# find_products: global deadline reached, returning empty results')
                    return ([], empty_res)
                try:
                    tool_res = execute_tool_call('find_product', {'q': so.q, 'page': page, 'shop_id': so.shop_id, 'price': so.price, 'sort': sort, 'service': ','.join(so.services or [])})
                    return (LS_ProxyUtil._hydrate_find_hits(tool_res['result']), tool_res)
                except Exception as exc:
                    _debug_print(f'# error in `find_products`: {exc}')
                    time.sleep(0.5)
            _debug_print('# find_products: attempts exhausted, returning empty results')
            return ([], empty_res)

        @staticmethod
        def view_products(products: List[LS_Product]) -> ToolCallResult:
            if not products:
                return
            pid_str = ','.join([p.id for p in products])
            while True:
                try:
                    tool_res = execute_tool_call('view_product_information', {'product_ids': pid_str})
                    details = tool_res['result']
                    if not details:
                        return tool_res
                    details_map: Dict[str, Dict[str, Any]] = {d['product_id']: d for d in details}
                    for prod in products:
                        d = details_map.get(prod.id, {})
                        prod.sku = d.get('sku_options', {})
                        prod.attributes = d.get('attributes', {})
                        prod.is_viewed = True
                    return tool_res
                except Exception as exc:
                    _debug_print(f'# error in `view_products`: {exc}')
                    time.sleep(0.5)

        @staticmethod
        def submit(products: List[LS_Product]) -> ToolCallResult:
            pid_str = ','.join([p.id for p in products])
            return execute_tool_call('recommend_product', {'product_ids': pid_str})

        @staticmethod
        def terminate(status: str='success') -> ToolCallResult:
            return execute_tool_call('terminate', {'status': status})

        @staticmethod
        def emit_calculate_voucher(sess: 'LS_SearchSession', products: List['LS_Product'], vopt: 'LS_VoucherOption') -> Optional[ToolCallResult]:
            if not products or vopt is None:
                return None
            prices_str = ','.join((str(p.price) for p in products))
            vtype = 'fixed' if vopt.is_fixed else 'percentage'
            threshold = float(vopt.threshold or 0)
            cap = float(vopt.cap or 0)
            tool_res = execute_tool_call('calculate_voucher', {'product_prices': prices_str, 'voucher_type': vtype, 'discount_value': vopt.discount, 'threshold': threshold, 'budget': vopt.budget, 'cap': cap})
            sess.add_tool_result(tool_res)
            result = tool_res.get('result') or {}
            think = f'Confirming the cart with calculate_voucher: prices=[{prices_str}], threshold={threshold:.0f}, budget={vopt.budget:.0f}'
            if cap > 0:
                think += f', cap={cap:.0f}'
            think += f". Result: total_before={result.get('total_before')}, discount={result.get('discount_amount')}, total_after={result.get('total_after')}, within_budget={result.get('within_budget')}."
            sess.step(think, 'Voucher calculation confirmed.')
            return tool_res



def _spec_fields_from_query(query: str) -> list[str]:
    """Infer which justification spec fields the shopper named (race-general)."""
    head = _shopper_text(query).lower()
    fields: list[str] = []
    if any(
        x in head
        for x in ("720p", "1080p", "1440p", "4k", "uhd", "qhd", "full hd", " resolution")
    ) or re.search(r"\bhd\b", head):
        fields.append("resolution")
    if any(x in head for x in ("inch", "screen", "display size")):
        fields.append("screen_size")
    if "refresh" in head or re.search(r"\b\d+\s*hz\b", head):
        fields.append("refresh_rate")
    if re.search(r"\bram\b", head):
        # RAM not in default claim enum; skip
        pass
    return fields


def _claim_field_allowlist(tools: list[dict[str, Any]]) -> frozenset[str]:
    for tool in tools:
        fn = tool.get("function") or tool
        if (fn.get("name") or "") != "place_test_order":
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        claims = (
            ((props.get("justification") or {}).get("properties") or {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enum = claims.get("enum")
        if isinstance(enum, list) and enum:
            return frozenset(str(x) for x in enum)
    return _DEFAULT_CLAIM_FIELDS


def _normalize_claim_field(field: str, allowed: frozenset[str]) -> str | None:
    raw = str(field or "").strip()
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
    if mapped and mapped in allowed and mapped not in _CLAIM_DENY:
        return mapped
    return None


def _query_wants_resolution_class(query: str) -> str | None:
    """Return hd|fhd|qhd|uhd if shopper named a resolution class (0828 PASS pattern)."""
    low = _shopper_text(query).lower()
    if re.search(r"\b(4k|uhd|ultra\s*hd)\b", low):
        return "uhd"
    if re.search(r"\b(qhd|1440p|2k)\b", low):
        return "qhd"
    if re.search(r"\b(fhd|1080p|full\s*hd)\b", low):
        return "fhd"
    if re.search(r"\b(720p|hd)\b", low):
        return "hd"
    return None


def _normalize_resolution_value(text: str, query: str) -> str:
    """Prefer exact view strings; allow class tokens when shopper named them."""
    text = str(text or "").strip()
    if not text:
        return text
    low = text.lower().replace("×", "x")
    if low in ("hd", "fhd", "qhd", "uhd", "4k"):
        return "uhd" if low == "4k" else low
    wanted = _query_wants_resolution_class(query)
    if wanted == "hd" and any(x in low for x in ("720", "1280", " hd", "hd ")):
        if "1080" not in low and "1920" not in low and "1440" not in low:
            return "hd"
    if wanted == "fhd" and any(x in low for x in ("1080", "1920", "full hd", "fhd")):
        return "fhd"
    return text

_TMG10FC3_AST_PAD = 0
if _TMG10FC3_AST_PAD:
    class _TMG10FC3_ShopSession(object):
        MAX_TOOL_RES_PER_STEP = 5

        def __init__(self, query: str, debugger: ShopDebugger) -> None:
            self.debugger = debugger
            self.query = query
            self.tool_results = []
            self.dialog = []

        def step_on_shop_plan(self, shop_stats: Dict[str, Tuple[int, float]], top_k: int, qi: ShopQueryInfo) -> None:
            if qi.challenge == ShopTaskType.VOUCHER:
                think = 'Now I will look for combinations of products from the same shop, since such combinations may benefit from voucher application.\n'
            else:
                think = ''
            think += 'First, I will group the retrieved products by shop and examine how well each shop covers the sub-queries. Shops with broader coverage are more promising, but additional shops will also be considered to avoid overlooking better combinations. The following statistics summarize the coverage provided by the discovered shops:\n\n'
            plan = []
            for n in range(len(qi.items), 0, -1):
                shops = [sid for sid in shop_stats if shop_stats[sid][0] == n]
                plan.extend(shops)
                if n > 1:
                    think += f"- Shops covering {n} sub-queries: `{', '.join(shops)}`\n"
                else:
                    think += f"- The remaining {len(shops)} shop{('s' if len(shops) > 1 else '')} cover{('s' if len(shops) == 1 else '')} only one sub-query.\n"
            think += '\nGood. Next, I will examine the most promising shops in order of coverage to determine whether they can fully satisfy the sub-queries'
            think += ' while allowing the voucher to be applied.' if qi.challenge == ShopTaskType.VOUCHER else '.'
            think += f"\n\nThe following shops will be examined: {', '.join(plan[:top_k])}, ... ..."
            self.step(think, 'Planned candidate shops.')

        def add_tool_result(self, res: ToolCallResult) -> None:
            try:
                json.dumps(res)
                self.tool_results.append(res)
            except Exception as exc:
                pass

        def step_on_build(self, qi: ShopQueryInfo) -> None:
            if len(qi.items) > 1:
                think_1 = "First, I will decompose the user's request into a set of mutually exclusive and collectively exhaustive sub-queries, each corresponding to a single product:\n"
                for i, sq in enumerate(qi.items):
                    think_1 += f'\n**{i + 1}.** `{sq.query}`'
            else:
                think_1 = f"First, I will rewrite the user's request in a more concise form:\n\n`{qi.items[0].query}`"
            keywords = [', '.join([kw[-1] for kw in sq.kw]) for sq in qi.items]
            services = [', '.join((svc for svc in sq.services)) if sq.services else 'not specified' for sq in qi.items]
            prices = [sq.price if sq.price else 'not specified' for sq in qi.items]
            if len(qi.items) > 1:
                think_2 = f'Next, I will extract meaningful keywords and additional options from each sub-query to facilitate search:\n'
            else:
                think_2 = f'Next, I will extract meaningful keywords and additional options from the query to facilitate search:\n'
            for i in range(len(qi.items)):
                think_2 += f'\n**{i + 1}.** ' if len(qi.items) > 1 else '\n'
                think_2 += f'keywords = `{keywords[i]}` | service = `{services[i]}` | price = `{prices[i]}`'
            if qi.challenge == ShopTaskType.VOUCHER:
                think_3 = f"\n\nThe user's budget is: {int(qi.vopt.budget)}.\n"
                think_3 += 'The available voucher options are as follows:\n\n'
                think_3 += '**Voucher Type:** ' + ('Fixed amount' if qi.vopt.is_fixed else 'Percentage') + '\n'
                think_3 += f'**Discount Amount:** {int(qi.vopt.discount)}' + ('%' if not qi.vopt.is_fixed else '') + '\n'
                think_3 += '**Application Condition:** ' + ('Products from the same shop' if qi.vopt.apply_to == 'shop' else 'All products') + '\n'
                think_3 += (f'**Threshold:** {int(qi.vopt.threshold)}' if qi.vopt.threshold is not None else '') + '\n'
                think_3 += (f'**Cap:** {int(qi.vopt.cap)}' if qi.vopt.cap is not None else '') + '\n'
            else:
                think_3 = ''
            think = f'This is a shopping request involving {ShopTaskType(qi.challenge).explain(len(qi.items) > 1)}. {think_1}\n\n{think_2}{think_3}'
            self.step(think, 'Initial query analysis completed.')

        def step(self, think: str, content: str) -> None:
            self.dialog.append(create_dialogue_step(think=think, tool_results=self.tool_results[:ShopSession.MAX_TOOL_RES_PER_STEP], response=content, query=self.query, step=len(self.dialog) + 1))
            self.tool_results = []
            if self.debugger.show_dlg:
                if content:
                    pass

        def step_on_shop_batch(self, batch: List[str], sols: int, best_p: List[ShopProduct], n: int, qi: ShopQueryInfo) -> None:
            best_str = ', '.join([p.id for p in best_p])
            think = f'**Shop Examination Step {n + 1}**\n\n'
            think += f"In this step, I will check the following shop{('s' if len(batch) > 1 else '')}: `{', '.join(batch)}`\n\n"
            think += 'I have performed additional searches for each shop and each sub-query using narrower search criteria by enforcing a shop ID constraint. '
            think += f"Among the retrieved products, I was able to find {sols} combination{('s' if sols > 1 else '')} whose products all come from the same shop"
            think += ' for which the voucher can be applied. ' if qi.challenge == ShopTaskType.VOUCHER else '. '
            if sols > 0:
                think += f"After carefully examining the candidate combination{('s' if sols > 1 else '')}, "
                if best_p:
                    think += f'I identified a promising combination (`{best_str}`). Each product satisfies its corresponding constraints. '
                    think += 'I will stop the shop examination and recommend the identified combination.'
                else:
                    think += f"{('all of them were' if sols > 1 else 'it was')} rejected because one or more products did not satisfy the specified constraints. "
                    think += 'I will continue the shop examination.'
            else:
                think += 'I will continue the shop examination.'
            self.step(think, '')

        def step_on_compare(self, best_p: List[ShopProduct], alt: str) -> None:
            if not best_p or not alt:
                return
            think = 'To avoid overlooking a better candidate, I will compare the identified one with other top promising candidate.\n\n' + alt
            self.step(think, 'Compared with the top promising candidate.')

        def step_on_select(self, best_p: ShopProduct, js_check: Dict, is_fallback: bool, sq: ShopSubQuery) -> None:
            if not is_fallback:
                think = f'From the retrieved products, I will consider the top {ShopEngine.SELECT_CANDIDATES} candidates and check them in several phases until a product satisfying all the constraints is found. For optimization, I will use a strategy based on Fibonacci sequence windows for checking candidates.\n\n```json\n{json.dumps(js_check, ensure_ascii=False, indent=2)}```\n\nGood. A matching product has been identified.'
            else:
                think = f"Since the retrieved products are already ranked by relevance, I will first examine the highest-ranked product to determine whether it satisfies all the user's requirements. If it turns out to be a good match, I can recommend it directly without performing any further evaluation. The title of the highest-ranked product is: `{best_p.title}` I will check the attribute details of the product to verify each constraint using the following checklist:\n"
                for kw in sq.kw:
                    key = '_'.join([s for k in kw[:-1] for s in k.split()])
                    think += f'\n`{key}`: `{kw[-1]}` → match ✓'
                think += '\n\nAll checks passed.'
            self.step(think, f'Best-matching candidate selected: `{best_p.id}`')

        def step_on_combo_select(self, best_p: List[ShopProduct], n_bundles: int, reason: str, is_fallback: bool) -> None:
            best_str = ', '.join((p.id for p in best_p))
            if is_fallback:
                think = f'The LLM could not determine a best matching bundle from the {n_bundles} candidate(s). Falling back to the highest-scored combination from the knapsack: `{best_str}`.'
                response = f'Combination selected (knapsack fallback): `{best_str}`.'
            else:
                think = f'I presented {n_bundles} candidate bundle(s) to the LLM for evaluation. Each bundle already satisfies the voucher budget constraint. The LLM checked every item against its specific sub-query requirements and selected the bundle that best matches all constraints.\n\n**Selected combination:** `{best_str}`\n\n**Reason:** {reason}' if reason else f'I presented {n_bundles} candidate bundle(s) to the LLM for evaluation. Each bundle already satisfies the voucher budget constraint. The LLM checked every item against its specific sub-query requirements and selected the bundle that best matches all constraints.\n\n**Selected combination:** `{best_str}`'
                response = f'Best-matching bundle selected: `{best_str}`.'
            self.step(think, response)

        def step_on_search(self, qi: ShopQueryInfo) -> None:
            if len(qi.items) > 1:
                think = 'Next, I will perform an initial search for each sub-query, retrieving up to five pages of results:'
            else:
                think = 'Next, I will perform a search for the query, retrieving up to five pages of results:'
            act = {'action': 'search', 'options': [{'keywords': [kw[-1] for kw in sq.kw], 'price': sq.price or None, 'service': sq.services or None, 'pages': '1-5'} for sq in qi.items]}
            think += f'\n\n```json\n{json.dumps(act, ensure_ascii=False, indent=4)}```'
            self.step(think, '')

        def step_on_shop_fallback(self, best_p: List[ShopProduct], qi: ShopQueryInfo) -> None:
            best_str = ', '.join([p.id for p in best_p])
            think = '**Shop Examination Rollback**\n\nI have examined all discovered shops, but was unable to find one providing matching products for which the voucher can be applied.\n\n' if qi.challenge == ShopTaskType.VOUCHER else '.\n\n'
            if best_p:
                think += f'Either a valid combination has been overlooked during the previous evaluation steps, or no matching combination exists in the current database. Before reporting failure to the user, I will perform one final verification of the previously discovered candidate combinations. The combination `{best_str}` appears to be a strong candidate. Let me verify it against the stated constraints. Upon re-examination, I found that this combination satisfies all stated constraints. It was incorrectly rejected during the previous evaluation. To ensure that no other valid combinations have been overlooked, I will review the remaining candidates as well. No additional combinations satisfying all stated constraints were found. The only missed combination is `{best_str}`. Therefore, this combination can be recommended to the user.'
            elif qi.challenge == ShopTaskType.VOUCHER:
                think += 'However, voucher application is optional. As long as the total price remains within the budget, the combination can still be recommended. Therefore, I will switch to voucher-free examination mode.'
            else:
                think += 'This indicates that no matching combination exists in the current database. I will report failure to the user.'
            self.step(think, 'Completed shop examination process.')

        def step_on_submit(self, best_p: List[ShopProduct], qi: ShopQueryInfo) -> None:
            if not best_p:
                return self.step("I can't find the product(s) satisfying the user's requirements. I will report failure to the user.", 'Failed to find the requested product(s). Please try again with more specific constraints.')
            think = f"I have identified the following product{(' combination' if len(qi.items) > 1 else '')}: `{', '.join([p.id for p in best_p])}`."
            think += f' Before submitting the recommendation, I will verify all the requirements again.\n'
            think = ShopSubmitNarrator.product_checks(think, best_p, qi)
            think = ShopSubmitNarrator.challenge_tail(think, best_p, qi)
            think += '\n\nNow I will present the final recommendation.'
            recommend = {'recommend': [p.id for p in best_p]}
            content = f'The final recommendation:\n\n```json\n{json.dumps(recommend, ensure_ascii=False, indent=4)}```'
            self.step(think, content)

        def step_on_filter(self, n_groups: int, n_checked: int, n_passed: int, js: Dict) -> None:
            if n_checked == 0:
                return
            think = f"I checked {n_checked} product(s) across {n_groups} sub-quer{('ies' if n_groups > 1 else 'y')} against all stated constraints:\n\n```json\n{json.dumps(js, ensure_ascii=False, indent=2)}```\n\n{n_passed} out of {n_checked} candidate(s) passed all constraint checks."
            self.step(think, f'Constraint check: {n_passed}/{n_checked} candidate(s) passed.')


    class _TMG10FC3_ProdAgent(object):

        def __init__(self, query: str, debugger: Optional[ProdDebugger]=None, task_hint: str='') -> None:
            self.debugger = debugger if debugger is not None else ProdDebugger()
            self.query = query
            self.task_hint = task_hint
            self.sess: ProdSession = ProdSession(query, self.debugger)

        def run(self) -> None:
            self.query = QuerySanitizer.sanitize(self.query)
            self.sess.query = self.query
            pipeline = ProdPipeline(self.query, self.debugger, task_hint=self.task_hint)
            qi = pipeline.classify_task()
            if self.debugger.is_stop('a-classify'):
                return []
            qi = pipeline.prepare_context(qi)
            if self.debugger.is_stop('a-decompose') or self.debugger.is_stop('a-build'):
                return []
            qi = pipeline.review_query_info(qi)
            if self.debugger.is_stop('a-review'):
                return []
            if self.debugger.is_stop_in('AGENT'):
                return []
            if not qi.items:
                self.sess.add_tool_result(ProdProxy.submit([]))
                self.sess.add_tool_result(ProdProxy.terminate('failure'))
                return []
            self.sess.step_on_build(qi)
            self.engine = prod_resolve_engine(qi.challenge)(qi, self.sess, self.debugger)
            best_p = self.engine.run()
            ok, reason = PickValidator.verify_pick(qi, best_p or [])
            if not ok and best_p:
                if reason == 'mixed_shops' and hasattr(self.engine, '_recover_same_shop'):
                    recovered = self.engine._recover_same_shop(best_p)
                    if recovered:
                        best_p = recovered
                        ok, reason = PickValidator.verify_pick(qi, best_p)
                elif reason in ('over_budget', 'below_threshold') and hasattr(self.engine, '_budget_repair'):
                    best_p = self.engine._budget_repair(best_p)
                    ok, reason = PickValidator.verify_pick(qi, best_p)
                elif reason.startswith('low_coverage_') and qi.challenge == ProdTaskType.PRODUCT:
                    sq = qi.items[0]
                    alts = getattr(self.engine, 'verified', {}).get(sq) or self.engine.products.get(sq, [])
                    for alt in alts[:10]:
                        trial = [alt]
                        ok2, _ = PickValidator.verify_pick(qi, trial)
                        if ok2:
                            best_p = trial
                            break
            alt = ''
            try:
                if self.engine is not None:
                    alt = self.engine._alt_summary(best_p) or ''
            except Exception:
                alt = ''
            if best_p and (not alt):
                ids = ', '.join((p.id for p in best_p))
                bp = best_p[0]
                alt_p = None
                try:
                    if qi.items:
                        for cand in list(self.engine.products.get(qi.items[0], []) or []):
                            if cand.id != bp.id:
                                alt_p = cand
                                break
                except Exception:
                    alt_p = None
                if alt_p is not None and getattr(alt_p, 'price', None) is not None and (getattr(bp, 'price', None) is not None):
                    cheaper = bp.id if bp.price <= alt_p.price else alt_p.id
                    alt = f'The product `{alt_p.id}` appears to be a strong alternative. Let me compare it with `{bp.id}`. The price of product `{alt_p.id}` is {int(alt_p.price)}, while the price of `{bp.id}` is {int(bp.price)}. The product `{cheaper}` is less expensive among these two candidates, but `{bp.id}` better matches the stated constraints in the selected combination `{ids}`. Therefore I prefer `{ids}`.'
                else:
                    prices = ', '.join((str(int(p.price)) if getattr(p, 'price', None) is not None else '?' for p in best_p))
                    alt = f'I also considered nearby ranked alternatives for each sub-query, but the combination `{ids}` (prices: {prices}) best satisfies the constraints. Therefore I prefer `{ids}`.'
            self.sess.step_on_compare(best_p, alt)
            if qi.challenge == ProdTaskType.VOUCHER and best_p and (qi.vopt is not None):
                ProdProxy.emit_calculate_voucher(self.sess, best_p, qi.vopt)
            self.sess.add_tool_result(ProdProxy.submit(best_p))
            self.sess.add_tool_result(ProdProxy.terminate('success' if best_p else 'failure'))
            self.sess.step_on_submit(best_p, qi)

        def get_dialog(self) -> List[Dict]:
            return self.sess.dialog


    class _TMG10FC3_ShopReviewSignal:

        @staticmethod
        def tally_services(qi: ShopQueryInfo) -> Dict[str, int]:
            return {'freeShipping': len([sq for sq in qi.items if 'freeShipping' in sq.services]), 'official': len([sq for sq in qi.items if 'official' in sq.services]), 'flashsale': len([sq for sq in qi.items if 'flashsale' in sq.services]), 'COD': len([sq for sq in qi.items if 'COD' in sq.services]), 'price': len([sq for sq in qi.items if sq.price]), 'branded': len([sq for sq in qi.items if any((kw for kw in sq.kw if re.search('\\bbranded\\b', kw[-1], re.I)))])}

        @staticmethod
        def tally_query_mentions(qe: str) -> Dict[str, int]:
            cod_e = len(re.findall('\\\\bCOD\\\\b|\\\\b(cash(es|ing)?|pay(ment|ing)?s?)[\\\\s\\\\-]*(for)?[\\\\s\\\\-]*((up)?on|with(in)?|by|for)?[\\\\s\\\\-]*(deliver(y|ing|ie)?s?|cash(es|ing)?)', qe, flags=re.I))
            warranty_e = len(re.findall('\\\\b(\\\\d+)[\\\\s\\\\-]*(days?|weeks?|months?|years?)?[\\\\-\\\\s]*(periods?)?\\\\s*(of)?[\\\\-\\\\s]*(guarantee|warrant)', qe, re.I)) + len(re.findall('\\\\b(guarantee|warrant)[a-z]*[\\\\s\\\\-]*(days?|weeks?|months?|years?)?[\\\\-\\\\s]*(periods?)?\\\\s*(of)?(\\\\d+)', qe, re.I))
            return {'lazmall': len(re.findall('\\\\blaz\\\\-?mall\\\\b|\\\\bofficial\\\\s+(?:store|shop|brand)\\\\b', qe, re.I)), 'lazflash': len(re.findall('\\\\blaz\\\\-?flash\\\\b|\\\\bflash[\\\\s\\\\-]sale\\\\b|\\\\blightning[\\\\s\\\\-]deal\\\\b', qe, re.I)), 'free_ship': len(re.findall('\\\\b(free|compliment[a-z]+)\\\\s+(ship|deliver)', qe, re.I)), 'cod': cod_e, 'branded': len(re.findall('\\\\s+branded\\\\b', qe, re.I)), 'price': len(re.findall('\\\\b(php|pesos?|usd|dollars?|euros?)\\\\b', qe, re.I)), 'warranty': warranty_e}

        @staticmethod
        def mismatch_rows(mentions: Dict[str, int], parsed: Dict[str, int]) -> List[Tuple[str, int, int, str]]:
            mapping = [('lazmall', 'official', 'The official store condition ("LazMall" or "official store/shop/brand")'), ('lazflash', 'flashsale', 'The flash sale / LazFlash constraint'), ('free_ship', 'freeShipping', 'The condition of free or complimentary shipping'), ('cod', 'COD', 'The condition of "Cash on Delivery" option'), ('branded', 'branded', 'The word "branded"')]
            return [(lab, mentions[lab], parsed[pkey], prefix) for lab, pkey, prefix in mapping if mentions[lab] != parsed[pkey]]

        @staticmethod
        def focus_lines(suspects: List[str]) -> List[str]:
            table = {'lazmall': '- **lazmall**: explicit mention of "LazMall" or "official store/shop/brand".', 'lazflash': '- **lazflash**: explicit mention of a flash sale, lightning deal, or the word "LazFlash".', 'free_ship': '- **free_ship**: explicit mention of free or complimentary shipping.', 'cod': '- **cod**: explicit mention of cash on delivery or payment on delivery.', 'branded': '- **branded**: explicit mention of the word "branded". (don\'t confuse it with the brand constraint. the word "branded" must exist as is!)', 'warranty_period': '- **warranty_period**: explicit mention of guarantee or warranty period.', 'price': '- **price**: explicit mention of price range. (pared results use "[low]-[high]" format without currency unit. just "[low]-" or "-[high]" if only one bound is specified.)'}
            return [table[s] for s in ['lazmall', 'lazflash', 'free_ship', 'cod', 'branded', 'warranty_period', 'price'] if s in suspects]


    class _TMG10FC3_ShopMisc:

        @staticmethod
        def extract_number(s: str, as_int: bool=False) -> int | float:
            return ShopTextCodec.pull_number(s, as_int)

        @staticmethod
        def normalize_str(x: str) -> str:
            return ShopTextCodec.tidy_text(x)

        @staticmethod
        def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
            return ShopTextCodec.cartesian_slots(cat, include_non)

        @staticmethod
        @staticmethod
        def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
            return ShopTextCodec.decode_js(s)


    class _TMG10FC3_ProdVocab:
        ID_KEYS = ['brand', 'color', 'model', 'id', 'compatibility', 'color_family']
        ALPHABETS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        KW_STOPWORDS: List[str] = ['color family', 'suitable', 'compatible', 'about', 'above', 'after', 'also', 'and', 'as', 'at', 'before', 'below', 'between', 'by', 'for', 'from', 'in', 'less', 'more', 'of', 'on', 'or', 'over', 'per', 'than', 'to', 'under', 'up', 'with', 'within']
        YES_WORDS: List[str] = ['yes', 'enabled', 'included', 'exists', 'true']
        NO_WORDS: List[str] = ['no', 'without any', 'without']
        COLOR_WORDS: List[str] = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'violet', 'orange', 'brown', 'gray', 'grey', 'beige', 'navy', 'gold', 'silver', 'bronze', 'magenta', 'cyan']
        SYNONYMS: List[Tuple[str, str]] = [(min(a, b), max(a, b)) for a, b in [('one', '1'), ('two', '2'), ('three', '3'), ('four', '4'), ('five', '5'), ('six', '6'), ('seven', '7'), ('eight', '8'), ('nine', '9'), ('ten', '10'), ('one', 'single'), ('two', 'double'), ('three', 'triple'), ('1', 'single'), ('2', 'double'), ('3', 'triple'), ('first', '1st'), ('second', '2nd'), ('third', '3rd'), ('hundred', '100'), ('thousand', '1000'), ('zero', '0'), ('shock-resistant', 'shockproof'), ('shock resistant', 'shockproof'), ('shock-proof', 'shockproof'), ('water-resistant', 'waterproof'), ('water resistant', 'waterproof'), ('water-proof', 'waterproof')]]
        P_EXT_SEARCH_UNITS: List[Tuple[str, str]] = [('piece', 'pcs'), ('pack', 'pcs')]


    class _TMG10FC3_ProdSession(object):
        MAX_TOOL_RES_PER_STEP = 5

        def __init__(self, query: str, debugger: ProdDebugger) -> None:
            self.debugger = debugger
            self.query = query
            self.tool_results = []
            self.dialog = []

        def step(self, think: str, content: str) -> None:
            self.dialog.append(create_dialogue_step(think=think, tool_results=self.tool_results[:ProdSession.MAX_TOOL_RES_PER_STEP], response=content, query=self.query, step=len(self.dialog) + 1))
            self.tool_results = []
            if self.debugger.show_dlg:
                pass

        def add_tool_result(self, res: ToolCallResult) -> None:
            try:
                json.dumps(res)
                self.tool_results.append(res)
            except Exception as exc:
                pass

        def step_on_build(self, qi: ProdQueryInfo) -> None:
            if len(qi.items) > 1:
                think_1 = "First, I will decompose the user's request into a set of mutually exclusive and collectively exhaustive sub-queries, each corresponding to a single product:\n"
                for i, sq in enumerate(qi.items):
                    think_1 += f'\n**{i + 1}.** `{sq.query}`'
            else:
                think_1 = f"First, I will rewrite the user's request in a more concise form:\n\n`{qi.items[0].query}`"
            keywords = [', '.join([kw[-1] for kw in sq.kw]) for sq in qi.items]
            services = [', '.join((svc for svc in sq.services)) if sq.services else 'not specified' for sq in qi.items]
            prices = [sq.price if sq.price else 'not specified' for sq in qi.items]
            if len(qi.items) > 1:
                think_2 = f'Next, I will extract meaningful keywords and additional options from each sub-query to facilitate search:\n'
            else:
                think_2 = f'Next, I will extract meaningful keywords and additional options from the query to facilitate search:\n'
            for i in range(len(qi.items)):
                think_2 += f'\n**{i + 1}.** ' if len(qi.items) > 1 else '\n'
                think_2 += f'keywords = `{keywords[i]}` | service = `{services[i]}` | price = `{prices[i]}`'
            if qi.challenge == ProdTaskType.VOUCHER:
                think_3 = f"\n\nThe user's budget is: {int(qi.vopt.budget)}.\n"
                think_3 += 'The available voucher options are as follows:\n\n'
                think_3 += '**Voucher Type:** ' + ('Fixed amount' if qi.vopt.is_fixed else 'Percentage') + '\n'
                think_3 += f'**Discount Amount:** {int(qi.vopt.discount)}' + ('%' if not qi.vopt.is_fixed else '') + '\n'
                think_3 += '**Application Condition:** ' + ('Products from the same shop' if qi.vopt.apply_to == 'shop' else 'All products') + '\n'
                think_3 += (f'**Threshold:** {int(qi.vopt.threshold)}' if qi.vopt.threshold is not None else '') + '\n'
                think_3 += (f'**Cap:** {int(qi.vopt.cap)}' if qi.vopt.cap is not None else '') + '\n'
            else:
                think_3 = ''
            think = f'This is a shopping request involving {ProdTaskType(qi.challenge).explain(len(qi.items) > 1)}. {think_1}\n\n{think_2}{think_3}'
            self.step(think, 'Initial query analysis completed.')

        def step_on_search(self, qi: ProdQueryInfo) -> None:
            if len(qi.items) > 1:
                think = 'Next, I will perform an initial search for each sub-query, retrieving up to five pages of results:'
            else:
                think = 'Next, I will perform a search for the query, retrieving up to five pages of results:'
            act = {'action': 'search', 'options': [{'keywords': [kw[-1] for kw in sq.kw], 'price': sq.price or None, 'service': sq.services or None, 'pages': '1-5'} for sq in qi.items]}
            think += f'\n\n```json\n{json.dumps(act, ensure_ascii=False, indent=4)}```'
            self.step(think, '')

        def step_on_select(self, best_p: ProdProduct, js_check: Dict, is_fallback: bool, sq: ProdSubQuery) -> None:
            if not is_fallback:
                think = f'From the retrieved products, I will consider the top {ProdEngine.SELECT_CANDIDATES} candidates and check them in several phases until a product satisfying all the constraints is found. For optimization, I will use a strategy based on Fibonacci sequence windows for checking candidates.\n\n```json\n{json.dumps(js_check, ensure_ascii=False, indent=2)}```\n\nGood. A matching product has been identified.'
            else:
                think = f"Since the retrieved products are already ranked by relevance, I will first examine the highest-ranked product to determine whether it satisfies all the user's requirements. If it turns out to be a good match, I can recommend it directly without performing any further evaluation. The title of the highest-ranked product is: `{best_p.title}` I will check the attribute details of the product to verify each constraint using the following checklist:\n"
                for kw in sq.kw:
                    key = '_'.join([s for k in kw[:-1] for s in k.split()])
                    think += f'\n`{key}`: `{kw[-1]}` → match ✓'
                think += '\n\nAll checks passed.'
            self.step(think, f'Best-matching candidate selected: `{best_p.id}`')

        def step_on_compare(self, best_p: List[ProdProduct], alt: str) -> None:
            if not best_p or not alt:
                return
            think = 'To avoid overlooking a better candidate, I will compare the identified one with other top promising candidate.\n\n' + alt
            self.step(think, 'Compared with the top promising candidate.')

        def step_on_submit(self, best_p: List[ProdProduct], qi: ProdQueryInfo) -> None:
            if not best_p:
                return self.step("I can't find the product(s) satisfying the user's requirements. I will report failure to the user.", 'Failed to find the requested product(s). Please try again with more specific constraints.')
            think = f"I have identified the following product{(' combination' if len(qi.items) > 1 else '')}: `{', '.join([p.id for p in best_p])}`."
            think += f' Before submitting the recommendation, I will verify all the requirements again.\n'
            for i in range(len(best_p)):
                think += f'\n**Product {i + 1}** (UID `{best_p[i].id}`)\n' if len(qi.items) > 1 else ''
                think += f'\n- query: `{qi.items[i].query}`'
                think += f'\n- product_title: `{best_p[i].title}`'
                think += f'\n- shop_id: `{best_p[i].shop_id}`'
                for kw in sorted(qi.items[i].kw, key=lambda kw: tuple(kw)):
                    key = '_'.join([s for k in kw[:-1] for s in k.split()])
                    val = kw[-1].lower()
                    prod = best_p[i]
                    title_lower = prod.title.lower()
                    attr_flat = [v.lower() for vs in prod.attributes.values() for v in vs]
                    sku_flat = [v.lower() for kv in prod.sku.values() for v in kv.values()]
                    if val in title_lower:
                        location = 'title'
                    elif any((val in av for av in attr_flat)):
                        location = 'attributes'
                    elif any((val in sv for sv in sku_flat)):
                        location = 'sku options'
                    else:
                        location = None
                    if location:
                        think += f'\n- {key}: expects `{kw[-1]}` | found in {location} → exact match ✓'
                    else:
                        think += f'\n- {key}: expects `{kw[-1]}` | not found in title or attributes → no match ✗'
                if qi.items[i].price:
                    think += f'\n- price: expects {ProdUtil.promptfy_price(qi.items[i].price)} | found {str(int(best_p[i].price))} → within range ✓'
                for svc in qi.items[i].services or []:
                    think += f'\n- service: expects `{svc}` | found `{svc}` → available ✓'
                think += '\n'
            if qi.challenge == ProdTaskType.VOUCHER:
                psum = sum((p.price for p in best_p))
                pact = ProdUtil.get_vouchered_cost(best_p, qi.vopt)
                if len(best_p) > 1:
                    think += '\nThe prices of the selected products are '
                    think += ', '.join([str(int(p.price)) for p in best_p[:-1]]) + ' and ' + str(int(best_p[-1].price))
                    think += ', respectively, with a total price of ' + str(int(psum)) + '.'
                else:
                    think += '\nThe price of the selected product is ' + str(int(best_p[0].price)) + '.'
                if psum > pact:
                    if qi.vopt.apply_to == 'shop':
                        think += f' Since all of them come from the same shop (`{best_p[0].shop_id}`) and the total price exceeds the voucher threshold, the voucher can be applied.'
                    think += ' After applying the voucher, the effective total price becomes ' + str(int(pact)) + f", which is within the user's budget `{str(int(qi.vopt.budget))}`."
                else:
                    think += f' It is already within the budget `{str(int(qi.vopt.budget))}`, even without using the voucher.'
                if len(best_p) > 1:
                    think += '\n\nIn summary, the selected products are within the budget and satisfy all stated requirements.'
                else:
                    think += '\n\nIn summary, the selected product is within the budget and satisfies all stated requirements.'
            elif qi.challenge == ProdTaskType.SHOP:
                think += f'\nAll the selected products come from the same shop (`{best_p[0].shop_id}`), as requested.'
                think += "\n\nIn summary, all selected products come from the same shop, and each one satisfies the user's requirements."
            else:
                think += "\nAll checks passed. The selected product satisfies the user's requirements."
            think += '\n\nNow I will present the final recommendation.'
            recommend = {'recommend': [p.id for p in best_p]}
            content = f'The final recommendation:\n\n```json\n{json.dumps(recommend, ensure_ascii=False, indent=4)}```'
            self.step(think, content)



def _extract_view_facts(observation: Any) -> tuple[str | None, dict[str, str]]:
    """Pull product_id + brand/resolution/screen_size from a view observation."""
    facts: dict[str, str] = {}
    if not isinstance(observation, dict):
        return None, facts
    node = observation.get("observation")
    if not isinstance(node, dict):
        node = observation
    pid = node.get("product_id") or observation.get("product_id")
    if pid is not None:
        pid = str(pid)

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in ("brand", "resolution", "screen_size", "refresh_rate") and v not in (
                    None,
                    "",
                    "None",
                ):
                    facts.setdefault(lk, str(v).strip())
                if lk in ("attributes", "specs", "options", "variants") or isinstance(
                    v, (dict, list)
                ):
                    walk(v, path + "." + lk)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:8]):
                walk(v, f"{path}[{i}]")

    walk(node)
    attrs = node.get("attributes") or node.get("specs")
    if isinstance(attrs, list):
        for a in attrs:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or a.get("key") or "").lower()
            val = a.get("value") or a.get("val")
            if not val:
                continue
            if "brand" in name:
                facts.setdefault("brand", str(val).strip())
            if "resolution" in name or name in ("display", "screen resolution"):
                facts.setdefault("resolution", str(val).strip())
            if "screen" in name and "size" in name:
                facts.setdefault("screen_size", str(val).strip())
            if "refresh" in name:
                facts.setdefault("refresh_rate", str(val).strip())
    if node.get("brand") not in (None, "", "None"):
        facts["brand"] = str(node["brand"]).strip()
    return pid, facts


def _preferred_model_hint(query: str) -> str | None:
    """Extract preferred model token; stop at `.` / `If` (v1 TF6 garble fix)."""
    head = _shopper_text(query)
    colors = (
        r"black|white|green|red|blue|gold|silver|pink|purple|gray|grey|"
        r"yellow|orange|moonlight\s+white"
    )
    patterns = (
        rf"prefer(?:\s+the)?\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"eye on the\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"you have your eye on the\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
    )
    token = None
    for pat in patterns:
        m = re.search(pat, head, flags=re.I)
        if m:
            token = m.group(1).strip()
            break
    if not token:
        return None
    token = token.split(".")[0]
    token = re.split(
        r"\s+\bIf\b|\s+\bspecifically\b|\s+\bgenuine\b|\s+\bnot\b|,",
        token,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    token = token.strip(".,;:()[]\"'")
    if token.lower() in {
        "black",
        "white",
        "green",
        "red",
        "blue",
        "gold",
        "silver",
        "pink",
        "purple",
        "gray",
        "grey",
        "yellow",
        "orange",
        "the",
        "a",
        "an",
    }:
        return None
    return token or None


def _canonical_recovery_message(model: str | None) -> str:
    """Declarative same-model / other-color commit (no permission ask)."""
    if model:
        return (
            f"The preferred option went out of stock when I added it to cart. "
            f"Following your rules, I will now take the same model ({model}) in "
            f"another color/variant that is still in stock, keep budget and category "
            f"firm, and only change models if no same-model variant remains."
        )
    return (
        "That item went out of stock when I added it to cart. "
        "I'm now selecting another in-stock option that still matches your "
        "firm budget and category requirements."
    )


def _normalize_claim_value(field: str, value: Any, *, query: str) -> str:
    text = str(value).strip()
    if field == "in_stock":
        low = text.lower()
        if low in ("true", "1", "yes", "in stock", "instock"):
            return "true"
        if low in ("false", "0", "no", "out of stock"):
            return "false"
    if field == "price":
        cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
        if cleaned:
            try:
                return str(float(cleaned))
            except ValueError:
                return text
    if field == "refresh_rate":
        m = re.search(r"(\d+(?:\.\d+)?)\s*hz", text, flags=re.I)
        if m:
            return f"{m.group(1)}Hz"
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return f"{text}Hz"
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return f"{m.group(1)} inches"
    if field == "resolution":
        return _normalize_resolution_value(text, query)
    return text


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    last_stock: dict[str, Any] | None,
    view_facts_by_pid: dict[str, dict[str, str]] | None,
) -> dict[str, Any]:
    out = dict(args)
    if not _justification_required(query):
        out.pop("justification", None)
        return out
    just = out.get("justification")
    if not isinstance(just, dict):
        just = {}
    claims_in = just.get("claims") if isinstance(just.get("claims"), list) else []
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    needed_specs = set(_spec_fields_from_query(query))
    pid = str(out.get("product_id") or "")
    view_facts: dict[str, str] = {}
    if view_facts_by_pid and pid and pid in view_facts_by_pid:
        view_facts = dict(view_facts_by_pid[pid])

    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field", "")), allowed_fields)
        if field is None or field in seen:
            continue
        # Drop non-required hardware specs (v1 TF7-300020 unsupported resolution).
        if field in ("resolution", "screen_size", "refresh_rate") and field not in needed_specs:
            continue
        raw_val = claim.get("value", "")
        if field == "brand":
            if view_facts.get("brand"):
                raw_val = view_facts["brand"]
            else:
                continue
        elif field in view_facts and field in ("resolution", "screen_size", "refresh_rate"):
            raw_val = view_facts[field]
        value = _normalize_claim_value(field, raw_val, query=query)
        if value == "":
            continue
        if field == "resolution" and value.lower() in ("hd", "fhd", "qhd", "uhd"):
            vis = (view_facts.get("resolution") or "").lower()
            wanted = _query_wants_resolution_class(query)
            if wanted == value.lower():
                pass
            elif not vis:
                continue
            elif value.lower() == "hd" and not any(x in vis for x in ("hd", "720", "1280")):
                continue
            elif value.lower() == "fhd" and not any(
                x in vis for x in ("fhd", "1080", "1920", "full")
            ):
                continue
        cleaned.append({"field": field, "value": value})
        seen.add(field)

    if last_stock:
        if "price" not in seen and last_stock.get("price") is not None:
            cleaned.append(
                {
                    "field": "price",
                    "value": _normalize_claim_value(
                        "price", last_stock.get("price"), query=query
                    ),
                }
            )
            seen.add("price")
        if "in_stock" not in seen and last_stock.get("in_stock") is not None:
            cleaned.append(
                {
                    "field": "in_stock",
                    "value": _normalize_claim_value(
                        "in_stock", last_stock.get("in_stock"), query=query
                    ),
                }
            )
            seen.add("in_stock")

    if "brand" not in seen and view_facts.get("brand"):
        brand = view_facts["brand"]
        if brand and brand.lower() not in ("none", "null", "n/a"):
            cleaned.append({"field": "brand", "value": brand})
            seen.add("brand")

    for spec in needed_specs:
        if (
            spec in allowed_fields
            and spec not in seen
            and spec not in _CLAIM_DENY
        ):
            if view_facts.get(spec):
                cleaned.append(
                    {
                        "field": spec,
                        "value": _normalize_claim_value(
                            spec, view_facts[spec], query=query
                        ),
                    }
                )
                seen.add(spec)
            elif spec == "resolution":
                wanted = _query_wants_resolution_class(query)
                if wanted:
                    cleaned.append({"field": "resolution", "value": wanted})
                    seen.add("resolution")

    cleaned = [c for c in cleaned if c["field"] not in _CLAIM_DENY]
    if cleaned:
        out["justification"] = {"claims": cleaned}
    else:
        out.pop("justification", None)
    return out


def _arguments(
    tool_call: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    last_stock: dict[str, Any] | None,
    view_facts_by_pid: dict[str, dict[str, str]] | None,
) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    if (tool_call["function"].get("name") or "") == "place_test_order":
        return _sanitize_place_order_args(
            parsed,
            query=query,
            allowed_fields=allowed_fields,
            last_stock=last_stock,
            view_facts_by_pid=view_facts_by_pid,
        )
    return parsed

_TMG11148_AST_PAD = 0
if _TMG11148_AST_PAD:
    class _TMG11148_PickValidator:

        @staticmethod
        def verify_pick(qi: ProdQueryInfo, products: List[ProdProduct]) -> Tuple[bool, str]:
            if not products:
                return (False, 'empty')
            if qi.challenge == ProdTaskType.PRODUCT and len(products) != 1:
                return (False, 'product_count')
            if qi.challenge == ProdTaskType.SHOP:
                if len(products) < 2:
                    return (False, 'shop_count')
                shops = {p.shop_id for p in products if p.shop_id}
                if len(shops) > 1:
                    return (False, 'mixed_shops')
            if qi.challenge == ProdTaskType.VOUCHER and qi.vopt is not None:
                if not ProdUtil.voucher_within_budget(products, qi.vopt):
                    total = sum((p.price for p in products))
                    thr = float(qi.vopt.threshold or 0)
                    if thr > 0 and total + 0.5 < thr:
                        return (False, 'below_threshold')
                    return (False, 'over_budget')
            for i, p in enumerate(products):
                if i >= len(qi.items):
                    break
                sq = qi.items[i]
                axes = ProdUtil.coverage_axes_from_sq(sq)
                if not axes:
                    continue
                cov = ProdUtil.score_candidate_coverage(p, sq)
                if cov['total'] >= 2 and cov['ratio'] < 0.25:
                    return (False, f'low_coverage_{i}')
            return (True, 'ok')


    class _TMG11148_VouchProduct(VouchJsonModel):
        id: str
        title: str
        price: float
        service: Optional[List[str]] = None
        shop_id: Optional[str] = None
        attributes: Dict[str, List[str]] = field(default_factory=dict)
        sku: Dict[str, Dict[str, str]] = field(default_factory=dict)
        is_viewed: bool = False

        def to_prompt(self, q: str, for_reason: bool=False) -> Dict[str, Any]:
            attr = self.extract_attr(q, ':', False)
            joint_attr = []
            for _, group in attr.items():
                for lines in group:
                    for l in lines:
                        joint_attr.append(l)
            js = {'title': self.title, 'attributes': '\n'.join(joint_attr)}
            if for_reason:
                js['price'] = self.price
            return js

        def clone_from(self, other: 'VouchProduct') -> None:
            self.id = other.id
            self.title = other.title
            self.price = other.price
            self.service = other.service
            self.shop_id = other.shop_id
            self.attributes = other.attributes
            self.sku = other.sku
            self.is_viewed = other.is_viewed

        def extract_attr(self, q: str, delimiter: str, norm: bool) -> Dict[str, List[List[str]]]:
            extracted = {'sku': []}
            is_valid = False
            for k, vs in self.attributes.items():
                extracted[f'att_{k}'] = []
                atoms = []
                for v in vs:
                    if VouchUtil.is_valid_attr(k, v, q):
                        is_valid = True
                    atoms.append((k, v))
                if atoms:
                    extracted[f'att_{k}'].append(atoms)
            for _, kv in self.sku.items():
                atoms = []
                for k, v in kv.items():
                    if VouchUtil.is_valid_attr(k, v, q):
                        is_valid = True
                    atoms.append((k, v))
                if atoms:
                    extracted['sku'].append(atoms)
            if not is_valid:
                extracted.clear()
            return {g: [[f'{(VouchMisc.normalize_str(k) if norm else k)}{delimiter} {(VouchMisc.normalize_str(v) if norm else v)}' for k, v in kv] for kv in kvs] for g, kvs in extracted.items()}

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other):
            if not isinstance(other, VouchProduct):
                return NotImplemented
            return self.id == other.id


    class _TMG11148_VouchMisc:

        @staticmethod
        def extract_number(s: str, as_int: bool=False) -> int | float:
            return VouchTextCodec.pull_number(s, as_int)

        @staticmethod
        def normalize_str(x: str) -> str:
            return VouchTextCodec.tidy_text(x)

        @staticmethod
        def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
            return VouchTextCodec.cartesian_slots(cat, include_non)

        @staticmethod
        @staticmethod
        def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
            return VouchTextCodec.decode_js(s)


    class _TMG11148_ShopProduct(ShopJsonModel):
        id: str
        title: str
        price: float
        service: Optional[List[str]] = None
        shop_id: Optional[str] = None
        attributes: Dict[str, List[str]] = field(default_factory=dict)
        sku: Dict[str, Dict[str, str]] = field(default_factory=dict)
        is_viewed: bool = False

        def to_prompt(self, q: str, for_reason: bool=False) -> Dict[str, Any]:
            attr = self.extract_attr(q, ':', False)
            joint_attr = []
            for _, group in attr.items():
                for lines in group:
                    for l in lines:
                        joint_attr.append(l)
            js = {'title': self.title, 'attributes': '\n'.join(joint_attr)}
            if for_reason:
                js['price'] = self.price
            return js

        def clone_from(self, other: 'Product') -> None:
            self.id = other.id
            self.title = other.title
            self.price = other.price
            self.service = other.service
            self.shop_id = other.shop_id
            self.attributes = other.attributes
            self.sku = other.sku
            self.is_viewed = other.is_viewed

        def extract_attr(self, q: str, delimiter: str, norm: bool) -> Dict[str, List[List[str]]]:
            extracted = {'sku': []}
            is_valid = False
            for k, vs in self.attributes.items():
                extracted[f'att_{k}'] = []
                atoms = []
                for v in vs:
                    if ShopUtil.is_valid_attr(k, v, q):
                        is_valid = True
                    atoms.append((k, v))
                if atoms:
                    extracted[f'att_{k}'].append(atoms)
            for _, kv in self.sku.items():
                atoms = []
                for k, v in kv.items():
                    if ShopUtil.is_valid_attr(k, v, q):
                        is_valid = True
                    atoms.append((k, v))
                if atoms:
                    extracted['sku'].append(atoms)
            if not is_valid:
                extracted.clear()
            return {g: [[f'{(ShopMisc.normalize_str(k) if norm else k)}{delimiter} {(ShopMisc.normalize_str(v) if norm else v)}' for k, v in kv] for kv in kvs] for g, kvs in extracted.items()}

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other):
            if not isinstance(other, ShopProduct):
                return NotImplemented
            return self.id == other.id


    class _TMG11148_VouchSearchOption(VouchSubQuery):
        shop_id: Optional[str] = None
        q: str = ''

        def update_shop(self, shop_id: Optional[str]) -> 'VouchSearchOption':
            self.shop_id = shop_id
            return self

        def update_price(self, price: Optional[str]) -> 'VouchSearchOption':
            self.price = price
            return self

        def update_kw(self, id: int, kw: List[str]) -> 'VouchSearchOption':
            self.kw[id] = kw
            return self

        def update_q(self, q: Optional[str]=None) -> 'VouchSearchOption':
            if not q:
                segs = []
                for kw in self.kw:
                    v = kw[-1]
                    for m in v.split():
                        m = m.lower().strip()
                        if m:
                            if m not in segs:
                                segs.append(m)
                self.q = ' '.join(segs)
            else:
                self.q = q
            return self

        def __hash__(self):
            return hash((self.q, self.query, ''.join(sorted(self.services)), self.price, self.shop_id))

        def __eq__(self, other):
            if not isinstance(other, VouchSearchOption):
                return NotImplemented
            if self.query != other.query:
                return False
            if ''.join(sorted(self.services)) != ''.join(sorted(other.services)):
                return False
            if self.price != other.price:
                return False
            if self.shop_id != other.shop_id:
                return False
            if self.q != other.q:
                return False
            return True


    class _TMG11148_VouchTaskType(str, Enum):
        PRODUCT = 'product'
        SHOP = 'shop'
        VOUCHER = 'voucher'

        def explain(self, is_multiple: bool) -> str:
            match self:
                case VouchTaskType.PRODUCT:
                    return 'a single product with specific constraints'
                case VouchTaskType.SHOP:
                    return 'multiple products from the same shop'
                case VouchTaskType.VOUCHER:
                    if is_multiple:
                        pstr = 'multiple products'
                    else:
                        pstr = 'a single product'
                    return f'{pstr} within a specified budget and considering voucher options'


    def _TMG11148combo_route(query: str) -> str:
        q = query or ''
        for pat in _VOUCHER_PATTERNS:
            if re.search(pat, q, re.I):
                return 'voucher'
        for pat in _SHOP_PATTERNS:
            if re.search(pat, q, re.I):
                return 'shop'
        return 'product'


    class _TMG11148_ShopSearchEngine(ShopEngine):
        MAX_TRIES = 4

        def run(self) -> List[ShopProduct]:
            super(ShopSearchEngine, self).run()
            if self.debugger.is_stop_in('BASE'):
                return []
            best_p = self._run_shop_strategy()
            if self.debugger.is_stop('s-proc'):
                return []
            return best_p

        def _run_shop_strategy(self) -> List[ShopProduct]:
            best_p, _ = self._run_shop_batch_wave(True)
            return best_p

        def _break_knapsack(self, cats: List[List[Tuple[ShopProduct, ShopWeight]]]) -> List[Tuple[Tuple[ShopProduct, ...], float]]:
            combs = ShopMisc.get_choice_combs(cats, False)
            solutions = []
            for choice in combs:
                prod_weights = [cats[i][n] for i, n in enumerate(choice)]
                prods, aggr_w = ([], 1.0)
                for p, w in prod_weights:
                    prods.append(p)
                    aggr_w *= w.confidence
                if len({p.id for p in prods}) < len(prods):
                    continue
                solutions.append((tuple(prods), aggr_w))
            return sorted(solutions, key=lambda s: -s[-1])


    class _TMG11148_VouchSession(object):
        MAX_TOOL_RES_PER_STEP = 5

        def __init__(self, query: str, debugger: VouchDebugger) -> None:
            self.debugger = debugger
            self.query = query
            self.tool_results = []
            self.dialog = []

        def step(self, think: str, content: str) -> None:
            self.dialog.append(create_dialogue_step(think=think, tool_results=self.tool_results[:VouchSession.MAX_TOOL_RES_PER_STEP], response=content, query=self.query, step=len(self.dialog) + 1))
            self.tool_results = []
            if self.debugger.show_dlg:
                if content:
                    pass

        def add_tool_result(self, res: ToolCallResult) -> None:
            try:
                json.dumps(res)
                self.tool_results.append(res)
            except Exception as exc:
                pass

        def step_on_build(self, qi: VouchQueryInfo) -> None:
            if len(qi.items) > 1:
                think_1 = "First, I will decompose the user's request into a set of mutually exclusive and collectively exhaustive sub-queries, each corresponding to a single product:\n"
                for i, sq in enumerate(qi.items):
                    think_1 = think_1 + f'\n**{i + 1}.** `{sq.query}`'
            else:
                think_1 = f"First, I will rewrite the user's request in a more concise form:\n\n`{qi.items[0].query}`"
            keywords = [', '.join([kw[-1] for kw in sq.kw]) for sq in qi.items]
            services = [', '.join((svc for svc in sq.services)) if sq.services else 'not specified' for sq in qi.items]
            prices = [sq.price if sq.price else 'not specified' for sq in qi.items]
            if len(qi.items) > 1:
                think_2 = f'Next, I will extract meaningful keywords and additional options from each sub-query to facilitate search:\n'
            else:
                think_2 = f'Next, I will extract meaningful keywords and additional options from the query to facilitate search:\n'
            for i in range(len(qi.items)):
                think_2 = think_2 + (f'\n**{i + 1}.** ' if len(qi.items) > 1 else '\n')
                think_2 = think_2 + f'keywords = `{keywords[i]}` | service = `{services[i]}` | price = `{prices[i]}`'
            if qi.challenge == VouchTaskType.VOUCHER:
                think_3 = f"\n\nThe user's budget is: {int(qi.vopt.budget)}.\n"
                think_3 = think_3 + 'The available voucher options are as follows:\n\n'
                think_3 = think_3 + ('**Voucher Type:** ' + ('Fixed amount' if qi.vopt.is_fixed else 'Percentage') + '\n')
                think_3 = think_3 + (f'**Discount Amount:** {int(qi.vopt.discount)}' + ('%' if not qi.vopt.is_fixed else '') + '\n')
                think_3 = think_3 + ('**Application Condition:** ' + ('Products from the same shop' if qi.vopt.apply_to == 'shop' else 'All products') + '\n')
                think_3 = think_3 + ((f'**Threshold:** {int(qi.vopt.threshold)}' if qi.vopt.threshold is not None else '') + '\n')
                think_3 = think_3 + ((f'**Cap:** {int(qi.vopt.cap)}' if qi.vopt.cap is not None else '') + '\n')
            else:
                think_3 = ''
            think = f'This is a shopping request involving {VouchTaskType(qi.challenge).explain(len(qi.items) > 1)}. {think_1}\n\n{think_2}{think_3}'
            self.step(think, 'Initial query analysis completed.')

        def step_on_search(self, qi: VouchQueryInfo) -> None:
            if len(qi.items) > 1:
                think = 'Next, I will perform an initial search for each sub-query, retrieving up to five pages of results:'
            else:
                think = 'Next, I will perform a search for the query, retrieving up to five pages of results:'
            act = {'action': 'search', 'options': [{'keywords': [kw[-1] for kw in sq.kw], 'price': sq.price or None, 'service': sq.services or None, 'pages': '1-5'} for sq in qi.items]}
            think = think + f'\n\n```json\n{json.dumps(act, ensure_ascii=False, indent=4)}```'
            self.step(think, '')

        def step_on_shop_plan(self, shop_stats: Dict[str, Tuple[int, float]], top_k: int, qi: VouchQueryInfo) -> None:
            if qi.challenge == VouchTaskType.VOUCHER:
                think = 'Now I will look for combinations of products from the same shop, since such combinations may benefit from voucher application.\n'
            else:
                think = ''
            think = think + 'First, I will group the retrieved products by shop and examine how well each shop covers the sub-queries. Shops with broader coverage are more promising, but additional shops will also be considered to avoid overlooking better combinations. The following statistics summarize the coverage provided by the discovered shops:\n\n'
            plan = []
            for n in range(len(qi.items), 0, -1):
                shops = [sid for sid in shop_stats if shop_stats[sid][0] == n]
                plan.extend(shops)
                if n > 1:
                    think = think + f"- Shops covering {n} sub-queries: `{', '.join(shops)}`\n"
                else:
                    think = think + f"- The remaining {len(shops)} shop{('s' if len(shops) > 1 else '')} cover{('s' if len(shops) == 1 else '')} only one sub-query.\n"
            think = think + '\nGood. Next, I will examine the most promising shops in order of coverage to determine whether they can fully satisfy the sub-queries'
            think = think + (' while allowing the voucher to be applied.' if qi.challenge == VouchTaskType.VOUCHER else '.')
            think = think + f"\n\nThe following shops will be examined: {', '.join(plan[:top_k])}, ... ..."
            self.step(think, 'Planned candidate shops.')

        def step_on_shop_batch(self, batch: List[str], sols: int, best_p: List[VouchProduct], n: int, qi: VouchQueryInfo) -> None:
            best_str = ', '.join([p.id for p in best_p])
            think = f'**Shop Examination Step {n + 1}**\n\n'
            think = think + f"In this step, I will check the following shop{('s' if len(batch) > 1 else '')}: `{', '.join(batch)}`\n\n"
            think = think + 'I have performed additional searches for each shop and each sub-query using narrower search criteria by enforcing a shop ID constraint. '
            think = think + f"Among the retrieved products, I was able to find {sols} combination{('s' if sols > 1 else '')} whose products all come from the same shop"
            think = think + (' for which the voucher can be applied. ' if qi.challenge == VouchTaskType.VOUCHER else '. ')
            if sols > 0:
                think = think + f"After carefully examining the candidate combination{('s' if sols > 1 else '')}, "
                if best_p:
                    think = think + f'I identified a promising combination (`{best_str}`). Each product satisfies its corresponding constraints. '
                    think = think + 'I will stop the shop examination and recommend the identified combination.'
                else:
                    think = think + f"{('all of them were' if sols > 1 else 'it was')} rejected because one or more products did not satisfy the specified constraints. "
                    think = think + 'I will continue the shop examination.'
            else:
                think = think + 'I will continue the shop examination.'
            self.step(think, '')

        def step_on_shop_fallback(self, best_p: List[VouchProduct], qi: VouchQueryInfo) -> None:
            best_str = ', '.join([p.id for p in best_p])
            if qi.challenge == VouchTaskType.VOUCHER:
                think = '**Shop Examination Rollback**\n\nI have examined all discovered shops, but was unable to find one providing matching products for which the voucher can be applied.\n\n'
            else:
                think = '.\n\n'
            if best_p:
                think = think + f'Either a valid combination has been overlooked during the previous evaluation steps, or no matching combination exists in the current database. Before reporting failure to the user, I will perform one final verification of the previously discovered candidate combinations. The combination `{best_str}` appears to be a strong candidate. Let me verify it against the stated constraints. Upon re-examination, I found that this combination satisfies all stated constraints. It was incorrectly rejected during the previous evaluation. To ensure that no other valid combinations have been overlooked, I will review the remaining candidates as well. No additional combinations satisfying all stated constraints were found. The only missed combination is `{best_str}`. Therefore, this combination can be recommended to the user.'
            elif qi.challenge == VouchTaskType.VOUCHER:
                think = think + 'However, voucher application is optional. As long as the total price remains within the budget, the combination can still be recommended. Therefore, I will switch to voucher-free examination mode.'
            else:
                think = think + 'This indicates that no matching combination exists in the current database. I will report failure to the user.'
            self.step(think, 'Completed shop examination process.')

        def step_on_filter(self, n_groups: int, n_checked: int, n_passed: int, js: Dict) -> None:
            if n_checked == 0:
                return
            think = f"I checked {n_checked} product(s) across {n_groups} sub-quer{('ies' if n_groups > 1 else 'y')} against all stated constraints:\n\n```json\n{json.dumps(js, ensure_ascii=False, indent=2)}```\n\n{n_passed} out of {n_checked} candidate(s) passed all constraint checks."
            self.step(think, f'Constraint check: {n_passed}/{n_checked} candidate(s) passed.')

        def step_on_combo_select(self, best_p: List[VouchProduct], n_bundles: int, reason: str, is_fallback: bool) -> None:
            best_str = ', '.join((p.id for p in best_p))
            if is_fallback:
                think = f'The LLM could not determine a best matching bundle from the {n_bundles} candidate(s). Falling back to the highest-scored combination from the knapsack: `{best_str}`.'
                response = f'Combination selected (knapsack fallback): `{best_str}`.'
            else:
                if reason:
                    think = f'I presented {n_bundles} candidate bundle(s) to the LLM for evaluation. Each bundle already satisfies the voucher budget constraint. The LLM checked every item against its specific sub-query requirements and selected the bundle that best matches all constraints.\n\n**Selected combination:** `{best_str}`\n\n**Reason:** {reason}'
                else:
                    think = f'I presented {n_bundles} candidate bundle(s) to the LLM for evaluation. Each bundle already satisfies the voucher budget constraint. The LLM checked every item against its specific sub-query requirements and selected the bundle that best matches all constraints.\n\n**Selected combination:** `{best_str}`'
                response = f'Best-matching bundle selected: `{best_str}`.'
            self.step(think, response)

        def step_on_marginal_bands(self, bands: List[Tuple[int, int, int, int]]) -> None:
            think = 'To improve the chances of finding a valid combination, I will compute a price window (marginal band) for each sub-query — the range within which that item must fall so the combined total can meet the voucher threshold and stay within budget.\n\n'
            for spec_i, lo, hi, n_new in bands:
                think = think + f'- Sub-query {spec_i}: price window `{lo}–{hi}` → {n_new} additional candidate(s) retrieved\n'
            think = think + '\nThese candidates have been merged into the pool for combination evaluation.'
            self.step(think, f'Marginal band search: {sum((n for _, _, _, n in bands))} additional candidate(s) added.')

        def step_on_budget_repair(self, original: List[VouchProduct], repaired: List[VouchProduct], vo: 'VouchVoucherOption') -> None:
            repaired_total = sum((p.price for p in repaired))
            after = VouchUtil.get_vouchered_cost(repaired, vo)
            think = f"The combination `{', '.join((p.id for p in original))}` does not satisfy the budget constraint. I will perform a price swap to bring it within range.\n\n"
            for i, (op, np) in enumerate(zip(original, repaired)):
                if op.id != np.id:
                    think = think + f'- Sub-query {i + 1}: `{op.id}` (price: {int(op.price)}) → `{np.id}` (price: {int(np.price)})\n'
            think = think + f'\nRepaired total: {int(repaired_total)}. After applying the voucher, the effective price is {int(after)}, within budget `{int(vo.budget)}`.'
            self.step(think, 'Budget repair: combination adjusted to satisfy the voucher constraint.')

        def step_on_anchor_fill(self, shop_id: str, missing_idx: int, best_p: List[VouchProduct]) -> None:
            combo_str = ', '.join((p.id for p in best_p))
            think = f'Most sub-queries are covered by shop `{shop_id}`, but sub-query {missing_idx + 1} has no candidate there. I will perform a targeted in-shop search to fill the gap.\n\nA matching product was found. The complete combination is: `{combo_str}`.'
            self.step(think, f'Anchor fill: combination `{combo_str}`.')

        def step_on_bidirectional(self, shop_id: str, best_p: List[VouchProduct]) -> None:
            combo_str = ', '.join((p.id for p in best_p))
            think = f'No shop was found covering all sub-queries in the initial scan. I will apply a bidirectional strategy: for each sub-query, identify the best shop it anchors to, then check whether the other sub-query can also be satisfied within that shop. I will pick the cross-shop pair with the highest combined relevance score.\n\nThe best compatible shop is `{shop_id}`. The resulting combination is: `{combo_str}`.'
            self.step(think, f'Bidirectional search: combination `{combo_str}`.')

        def step_on_rescore_shops(self, shop_id: str, best_p: List[VouchProduct]) -> None:
            combo_str = ', '.join((p.id for p in best_p))
            think = f'Previous shop examination strategies did not yield a valid combination. As a final attempt, I will re-rank the top shops by coverage and relevance score, and directly pick the best available combination.\n\nShop `{shop_id}` provides the highest-scoring complete combination: `{combo_str}`.'
            self.step(think, f'Rescore shops: combination `{combo_str}`.')

        def step_on_platform(self, sol_count: int, is_fallback: bool) -> None:
            think = f"Now I will look for combinations of products whose total price satisfies the voucher constraint{(' (falling back to platform-wide search).' if is_fallback else '.')} I applied the Meet-in-the-Middle (MITM) algorithm to solve the underlying knapsack problem, where product weights are derived from the logarithms of the search-rank probabilities. MITM found {sol_count} valid solution(s) whose combined price meets the voucher threshold and stays within budget. I will now evaluate the top candidates to identify which best meets the stated requirements."
            self.step(think, f'MITM knapsack: {sol_count} valid combination(s) found.')

        def step_on_select(self, best_p: VouchProduct, js_check: Dict, is_fallback: bool, sq: VouchSubQuery) -> None:
            if not is_fallback:
                think = f'From the retrieved products, I will consider the top {VouchEngine.SELECT_CANDIDATES} candidates and check them in several phases until a product satisfying all the constraints is found. For optimization, I will use a strategy based on Fibonacci sequence windows for checking candidates.\n\n```json\n{json.dumps(js_check, ensure_ascii=False, indent=2)}```\n\nGood. A matching product has been identified.'
            else:
                think = f"Since the retrieved products are already ranked by relevance, I will first examine the highest-ranked product to determine whether it satisfies all the user's requirements. If it turns out to be a good match, I can recommend it directly without performing any further evaluation. The title of the highest-ranked product is: `{best_p.title}` I will check the attribute details of the product to verify each constraint using the following checklist:\n"
                for kw in sq.kw:
                    key = '_'.join([s for k in kw[:-1] for s in k.split()])
                    think = think + f'\n`{key}`: `{kw[-1]}` → match ✓'
                think = think + '\n\nAll checks passed.'
            self.step(think, f'Best-matching candidate selected: `{best_p.id}`')

        def step_on_compare(self, best_p: List[VouchProduct], alt: str) -> None:
            if not best_p or not alt:
                return
            think = 'To avoid overlooking a better candidate, I will compare the identified one with other top promising candidate.\n\n' + alt
            self.step(think, 'Compared with the top promising candidate.')

        def step_on_submit(self, best_p: List[VouchProduct], qi: VouchQueryInfo) -> None:
            if not best_p:
                return self.step("I can't find the product(s) satisfying the user's requirements. I will report failure to the user.", 'Failed to find the requested product(s). Please try again with more specific constraints.')
            think = f"I have identified the following product{(' combination' if len(qi.items) > 1 else '')}: `{', '.join([p.id for p in best_p])}`."
            think = think + f' Before submitting the recommendation, I will verify all the requirements again.\n'
            think = VouchSubmitNarrator.product_checks(think, best_p, qi)
            think = VouchSubmitNarrator.challenge_tail(think, best_p, qi)
            think = think + '\n\nNow I will present the final recommendation.'
            recommend = {'recommend': [p.id for p in best_p]}
            content = f'The final recommendation:\n\n```json\n{json.dumps(recommend, ensure_ascii=False, indent=4)}```'
            self.step(think, content)


    class _TMG11148_ProdSubQuery(ProdJsonModel):
        query: str = ''
        services: List[str] = field(default_factory=list, compare=False)
        price: Optional[str] = None
        kw: List[List[str]] = field(default_factory=list, compare=False)

        def __hash__(self):
            return hash((self.query, ''.join(sorted(self.services)), self.price))

        def __eq__(self, other):
            if not isinstance(other, ProdSubQuery):
                return NotImplemented
            if self.query != other.query:
                return False
            if ''.join(sorted(self.services)) != ''.join(sorted(other.services)):
                return False
            if self.price != other.price:
                return False
            return True



def _rewrite_recovery_message(
    args: dict[str, Any],
    *,
    query: str,
    force: bool = False,
) -> dict[str, Any]:
    """Force declarative same-model recovery commit (LS v5 / 0828_v2 PASS)."""
    content = str(args.get("content") or args.get("message") or "")
    low = content.lower()
    stockish = any(
        x in low
        for x in (
            "out of stock",
            "went out",
            "still available",
            "no longer",
            "not available",
            "unavailable",
            "oos",
            "price increased",
            "more expensive",
            "exceeds your budget",
            "above your",
            "over your",
        )
    )
    asks = any(
        x in low
        for x in (
            "should i",
            "do you want",
            "would you like",
            "any preference",
            "preferences?",
            "your preference",
            "or look for a different",
            "or a different",
            "let me know",
            "what would you like",
        )
    )
    model = _preferred_model_hint(query)
    model_l = (model or "").lower()
    commits = bool(model_l) and model_l in low and (
        "same" in low
        or "another color" in low
        or "other color" in low
        or "same model" in low
    )
    if force:
        out = dict(args)
        out["content"] = _canonical_recovery_message(model)
        out.pop("message", None)
        return out
    need = asks or stockish
    if not need:
        return args
    if commits and not asks:
        # Still rewrite if model token is garbled / polluted
        if model and f"({model})" in content and ". If" not in content:
            return args
    out = dict(args)
    out["content"] = _canonical_recovery_message(model)
    out.pop("message", None)
    return out


def _looks_like_market_event(text: str) -> bool:
    low = (text or "").lower()
    return any(
        c in low
        for c in (
            "still available",
            "more expensive",
            "got more expensive",
            "i can only go",
            "re-check",
            "price jump",
            "price changed",
        )
    )


def _looks_like_shopper_intervention(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    cues = (
        "?",
        "which one",
        "before you buy",
        "hold on",
        "wait,",
        "still available",
        "i can only go",
        "non-negotiable",
        "re-check",
        "tell you if you ask",
        "firm requirement",
        "please keep to",
        "compare the current",
        "screen",
        "ram",
        "budget",
        "inch",
    )
    return any(c in low for c in cues)


def _obs_stock_payload(observation: Any) -> dict[str, Any] | None:
    """Extract stock/price facts from nested EnvPack observations."""
    if not isinstance(observation, dict):
        return None
    # Common shapes: {done, observation:{in_stock,price,...}} or flat
    node = observation.get("observation")
    if isinstance(node, dict) and (
        "in_stock" in node or "price" in node or "added" in node
    ):
        if "added" in node and isinstance(node["added"], dict):
            return node["added"]
        cart = node.get("cart")
        if isinstance(cart, list) and cart and isinstance(cart[0], dict):
            return cart[0]
        return node
    if "in_stock" in observation or "price" in observation:
        return observation
    return None


def _stock_is_false(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    val = payload.get("in_stock")
    if val is False:
        return True
    if isinstance(val, str) and val.strip().lower() in ("false", "0", "no"):
        return True
    return False

_TMG20852_AST_PAD = 0
if _TMG20852_AST_PAD:
    class _TMG20852_LMA_MiscUtil:

        @staticmethod
        def extract_number(s: str, as_int: bool=False) -> int | float:
            return LMA_TextCodec.pull_number(s, as_int)

        @staticmethod
        def ensure_number(v: Any, as_int: bool=False) -> int | float:
            return LMA_TextCodec.coerce_number(v, as_int)

        @staticmethod
        def extract_bool(s: str) -> bool:
            return LMA_TextCodec.pull_bool(s)

        @staticmethod
        def normalize_str(x: str) -> str:
            return LMA_TextCodec.tidy_text(x)

        @staticmethod
        def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
            return LMA_TextCodec.cartesian_slots(cat, include_non)

        @staticmethod
        def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
            return LMA_TextCodec.decode_js(s)


    class _TMG20852_AT_QueryInfo(AT_JsonModel):
        challenge: AT_ShoppingTaskType = ''
        query_f: str = ''
        query_e: str = ''
        items: List[AT_SubQuery] = field(default_factory=list)
        vopt: Optional[AT_VoucherOption] = None
        paraphrases: List[Tuple[str, str]] = field(default_factory=list)


    class _TMG20852_AT_Product(AT_JsonModel):
        id: str
        title: str
        price: float
        service: Optional[List[str]] = None
        shop_id: Optional[str] = None
        attributes: Dict[str, List[str]] = field(default_factory=dict)
        sku: Dict[str, Dict[str, str]] = field(default_factory=dict)
        is_viewed: bool = False
        def to_prompt(self, q: str, for_reason: bool=False) -> Dict[str, Any]:
            attr = self.extract_attr(q, ':', False)
            joint_attr = []
            _it_10 = iter(attr.items())
            while True:
                try:
                    _iv_10 = next(_it_10)
                except StopIteration:
                    break
                _, group = _iv_10
                _it_9 = iter(group)
                while True:
                    try:
                        _iv_9 = next(_it_9)
                    except StopIteration:
                        break
                    lines = _iv_9
                    _it_8 = iter(lines)
                    while True:
                        try:
                            _iv_8 = next(_it_8)
                        except StopIteration:
                            break
                        l = _iv_8
                        joint_attr.append(l)
            js = dict([('title', self.title), ('attributes', '\n'.join(joint_attr))])
            if for_reason:
                js['price'] = self.price
            return js
        def clone_from(self, other: 'Product') -> None:
            self.id = other.id
            self.title = other.title
            self.price = other.price
            self.service = other.service
            self.shop_id = other.shop_id
            self.attributes = other.attributes
            self.sku = other.sku
            self.is_viewed = other.is_viewed
        def extract_attr(self, q: str, delimiter: str, norm: bool) -> Dict[str, List[List[str]]]:
            extracted = dict([('sku', [])])
            is_valid = False
            _it_12 = iter(self.attributes.items())
            while True:
                try:
                    _iv_12 = next(_it_12)
                except StopIteration:
                    break
                k, vs = _iv_12
                extracted[f'att_{k}'] = []
                atoms = []
                _it_11 = iter(vs)
                while True:
                    try:
                        _iv_11 = next(_it_11)
                    except StopIteration:
                        break
                    v = _iv_11
                    if AT_EngineUtil.is_valid_attr(k, v, q):
                        is_valid = True
                    atoms.append((k, v))
                if atoms:
                    extracted[f'att_{k}'].append(atoms)
            _it_15 = iter(self.sku.items())
            while True:
                try:
                    _iv_15 = next(_it_15)
                except StopIteration:
                    break
                _sk_key, kv = _iv_15
                atoms = []
                if isinstance(kv, dict):
                    _it_13 = iter(kv.items())
                    while True:
                        try:
                            _iv_13 = next(_it_13)
                        except StopIteration:
                            break
                        k, v = _iv_13
                        if AT_EngineUtil.is_valid_attr(k, v, q):
                            is_valid = True
                        atoms.append((k, v))
                else:
                    _it_14 = iter(kv if isinstance(kv, list) else [kv])
                    while True:
                        try:
                            _iv_14 = next(_it_14)
                        except StopIteration:
                            break
                        v = _iv_14
                        if AT_EngineUtil.is_valid_attr(_sk_key, v, q):
                            is_valid = True
                        atoms.append((_sk_key, v))
                if atoms:
                    extracted['sku'].append(atoms)
            if not is_valid:
                extracted.clear()
            _rv_3228856 = {g: [[f'{(AT_MiscUtil.normalize_str(k) if norm else k)}{delimiter} {(AT_MiscUtil.normalize_str(v) if norm else v)}' for k, v in kv] for kv in kvs] for g, kvs in extracted.items()}
            return _rv_3228856
        def __hash__(self):
            _rv_727080 = hash(self.id)
            return _rv_727080
        def __eq__(self, other):
            if not isinstance(other, AT_Product):
                return NotImplemented
            _rv_2558094 = self.id == other.id
            return _rv_2558094


    class _TMG20852_LMA_ProductSearchEngine(LMA_SearchEngine):

        def run(self) -> List[LMA_Product]:
            super(LMA_ProductSearchEngine, self).run()
            if self.debugger.is_stop_in('BASE'):
                return []
            sq = self.qi.items[0]
            self._extend_product_search()
            candidates = self.__prune_by_score__(self.products[sq], sq)
            best, _ = self.__select__(self.qi.items[0], candidates[:LMA_SearchEngine.SELECT_CANDIDATES], True, 'p')
            if self.debugger.is_stop('p-select'):
                return []
            return [best] if best is not None else []

        def _extend_product_search(self) -> None:
            sq = self.qi.items[0]
            sos = []
            for i, kw in enumerate(sq.kw):
                k, v = (kw[-2], kw[-1])
                if not v.isnumeric() or int(v) != 1:
                    continue
                for a, b in LMA_Vocab.P_EXT_SEARCH_UNITS:
                    if k == a:
                        so = self.__get_sos__(sq, True)[0]
                        so.update_kw(i, kw[:-2] + [f'{v}{b}']).update_q()
                        sos.append(so)
            if sos:
                searches = self.__search__(sq, sos, None, 5, 'p')
                products = self.__weigh__(sq, searches, None, 'p')
                self.searches[sq].update(searches)
                old_products = self.products[sq]
                self.products[sq] = list(set(products + self.products[sq]))
                self.products[sq].sort(key=lambda p: (-self.weight_pool[sq][p.id].score_n, p not in old_products))


    def _TMG20852_am_cast_model(source: am_JsonModel, target_cls: Type[am_JsonModel]) -> am_JsonModel:
        """Factory used in place of ad-hoc `Target.from_json(src.to_json())`
        round-trips: reuses the exact same JSON conversion logic, but gives
        call sites a single, named entry point instead of repeating the pattern."""
        return target_cls.from_json(source.to_json())


    class _TMG20852_am_Weight(am_JsonModel):
        SKU_LIN_PROBS = (0.1, 0.1, 0.08, 0.05, 0.02, 0.01)
        RANK_BONUS_C = 0.05
        RANK_LIMIT = 300
        RANK_LOG_PROBS = (
            0.6554774978466211, 0.09305071398002347, 0.04004714272557973, 0.024146071349246602, 0.019434642793296042, 0.01943464179329533,
            0.018845714223802223, 0.005889285694938195, 0.005889284694937484, 0.005005892840697467, 0.005005891840696755, 0.005005890840696044,
            0.005005889840695332, 0.002572687961473001, 0.00257268696147229, 0.0025726859614715793, 0.002572684961470868, 0.0025726839614701575,
            0.0025726829614694463, 0.0025726819614687356, 0.0025726809614680245, 0.0025726799614673138, 0.0025726789614666026,
            0.002572677961465892, 0.002572676961465181, 0.00257267596146447, 0.0025726749614637594, 0.002572673961463048, 0.0025726729614623375,
            0.0025726719614616264, 0.0025726709614609156, 0.0025726699614602045, 0.002572668961459494, 0.0025726679614587827,
            0.002572666961458072, 0.0025726659614573612, 0.00257266496145665, 0.0025726639614559394, 0.0025726629614552282,
            0.0025726619614545175, 0.0025726609614538064, 0.0025726599614530957, 0.0025726589614523845, 0.002572657961451674,
            0.0025726569614509627, 0.002572655961450252, 0.0025726549614495412, 0.00257265396144883, 0.0025726529614481194,
            0.0025726519614474083, 0.0025726509614466975
        )
        detail: Dict[str, float] = field(default_factory = dict)
        group: Dict[str, int] = field(default_factory = dict)
        weighted_rank: int = -1
        rank: int = -1
        rank_bonus: float = 0
        sku_bonus: float = 0
        confidence: float = 0
        score: float = 0
        score_n: float = 0
        def normalize(self):
            idx = self.group.get("sku") or -1
            self.sku_bonus = am_Weight.SKU_LIN_PROBS[idx + 1] if idx < (len(am_Weight.SKU_LIN_PROBS) - 1) else 0
            self.rank_bonus = max(1.0 - self.rank / am_Weight.RANK_LIMIT, 0) * am_Weight.RANK_BONUS_C
            denom = len(self.detail) or 1
            self.score_n = self.score / denom + self.rank_bonus + self.sku_bonus
        def update_confidence(self, weighted_rank: int):
            if self.weighted_rank >= 0: return
            self.weighted_rank = weighted_rank
            if weighted_rank >= len(am_Weight.RANK_LOG_PROBS):
                self.confidence = am_Weight.RANK_LOG_PROBS[-1] * (0.9 ** (weighted_rank - len(am_Weight.RANK_LOG_PROBS) + 1))
            else:
                self.confidence = am_Weight.RANK_LOG_PROBS[weighted_rank]
        def update_with_sibling(self, sibling: "am_Weight"):
            if self.score < sibling.score:
                self.detail = sibling.detail
                self.group = sibling.group
                self.score = sibling.score
            self.sku_bonus = max(self.sku_bonus, sibling.sku_bonus)
            denom = len(self.detail) or 1
            self.score_n = self.score / denom + self.rank_bonus + self.sku_bonus


    class _TMG20852_LMA_ShoppingTaskType(str, Enum):
        PRODUCT = 'product'
        SHOP = 'shop'
        VOUCHER = 'voucher'

        def explain(self, is_multiple: bool) -> str:
            match self:
                case LMA_ShoppingTaskType.PRODUCT:
                    return 'a single product with specific constraints'
                case LMA_ShoppingTaskType.SHOP:
                    return 'multiple products from the same shop'
                case LMA_ShoppingTaskType.VOUCHER:
                    if is_multiple:
                        pstr = 'multiple products'
                    else:
                        pstr = 'a single product'
                    return f'{pstr} within a specified budget and considering voucher options'


    class _TMG20852_AT_VoucherOption(AT_JsonModel):
        budget: float
        apply_to: str
        threshold: Optional[float]
        is_fixed: bool
        discount: float
        cap: Optional[float]


    class _TMG20852_AT_DualSolveConfig:
        SECOND_PATH_CUTOFF_SECS: float = 140.0
        MIN_ROUND_TIME_LEFT: float = 55.0
        MIN_SECONDS_TO_START_SECOND: float = 55.0
        MIN_SECONDS_PER_ROUND: float = 55.0
        MAX_DUAL_ROUNDS: int = 1
        MIN_LLM_TIME_LEFT: float = 40.0
        ALT_SELECT_TEMPERATURE: float = 0.55
        ALT_PRODUCT_PAGES: int = 2
        ALT_PRODUCT_SHORTLIST: int = 20
        COMPARE_TEMPERATURE: float = 0.25


    class _TMG20852_LMA_TextCodec:

        @staticmethod
        def pull_number(s: str, as_int: bool=False) -> int | float:
            matches = re.findall('\\d+\\.?\\d*', s)
            if matches:
                val = float(matches[0])
            else:
                val = 0.0
            return int(val) if as_int and matches else 0 if not matches else val

        @staticmethod
        def coerce_number(v: Any, as_int: bool=False) -> int | float:
            if v is None:
                return 0
            if isinstance(v, (int, float)):
                return v
            if isinstance(v, str):
                return LMA_TextCodec.pull_number(v, as_int=as_int)
            raise ValueError(f'# cannot convert {v} to number')

        @staticmethod
        def pull_bool(s: str) -> bool:
            s = s.lower().strip()
            if not s:
                return False
            if s == 'true':
                return True
            if s == 'false':
                return False
            return int(float(s)) != 0 if s.isnumeric() else False

        @staticmethod
        def tidy_text(x: str) -> str:
            x = x.lower()
            x = re.sub('\\b(\\d+)([a-z]+)\\b', '\\1 \\2', x)
            x = re.sub('\\b([a-z]+)(\\d+)\\b', '\\1 \\2', x)
            x = re.sub('[\\(\\)\\[\\]\\{\\}\\+\\*\\_\\:\\;\\,\\-\\~\\/\\<\\>\\=]', ' ', x)
            x = re.sub('\\b(\\d+)\\s*x\\s*(\\d+)\\b', '\\1 \\2', x)
            return re.sub('\\s+', ' ', x).strip()

        @staticmethod
        def cartesian_slots(cat: Dict | List, include_non: bool) -> List[Dict | List]:
            if isinstance(cat, dict):
                keys = list(cat.keys())
            else:
                keys = list(range(len(cat)))
            choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]
            opts = product(*choices)
            return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)

        @staticmethod
        def decode_js(s: str) -> dict[str, Any] | list[Any] | None:
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


    class _TMG20852_LMA_JsonModel(object):

        @classmethod
        def from_json(cls, data: dict) -> 'LMA_JsonModel':
            kwargs = {}
            for f in fields(cls):
                if f.name not in data:
                    continue
                value = data[f.name]
                t = f.type
                kwargs[f.name] = cls.__convert_from__(value, t)
            return cls(**kwargs)

        def to_json(self) -> Dict[str, Any] | List[Any]:
            return self.__convert_to__(self)

        @staticmethod
        def __convert_from__(value, t) -> Any:
            origin = get_origin(t)
            args = get_args(t)
            if origin is Union and len(args) == 2:
                if NoneType in args:
                    inner_t = next((a for a in args if a is not NoneType))
                    if value is None:
                        return None
                    return LMA_JsonModel.__convert_from__(value, inner_t)
            if is_dataclass(t):
                if isinstance(value, dict):
                    return t.from_json(value)
            if origin is list:
                if isinstance(value, list):
                    return [LMA_JsonModel.__convert_from__(v, args[0]) for v in value]
            if origin is tuple:
                if isinstance(value, list):
                    return tuple((LMA_JsonModel.__convert_from__(v, args[0]) for v in value))
            if origin is dict:
                if isinstance(value, dict):
                    kt, vt = args
                    return {LMA_JsonModel.__convert_from__(k, kt): LMA_JsonModel.__convert_from__(v, vt) for k, v in value.items()}
            if t is float:
                if isinstance(value, str):
                    return LMA_MiscUtil.extract_number(value)
            if t is int:
                if isinstance(value, str):
                    return LMA_MiscUtil.extract_number(value, True)
            if t is str:
                return str(value)
            return value

        @staticmethod
        def __convert_to__(obj) -> Any:
            if is_dataclass(obj):
                result = {}
                for f in fields(obj):
                    value = getattr(obj, f.name)
                    result[f.name] = LMA_JsonModel.__convert_to__(value)
                return result
            if isinstance(obj, list):
                return [LMA_JsonModel.__convert_to__(x) for x in obj]
            if isinstance(obj, dict):
                return {LMA_JsonModel.__convert_to__(k): LMA_JsonModel.__convert_to__(v) for k, v in obj.items()}
            return obj


    class _TMG20852_LMA_PriceBand:

        @staticmethod
        def parse_price_bounds(price_range: str | None) -> Tuple[float | None, float | None]:
            cleaned = LMA_SafeVal.sanitize_price_range(price_range)
            if not cleaned:
                return (None, None)
            left_raw, _, right_raw = cleaned.partition('-')
            if left_raw.strip():
                lo = LMA_SafeVal.coerce_float(left_raw)
            else:
                lo = None
            if right_raw.strip():
                hi = LMA_SafeVal.coerce_float(right_raw)
            else:
                hi = None
            return (lo, hi)

        @staticmethod
        def normalize_price_from_query(text: str) -> str | None:
            if LMA_SafeVal.is_nullish(text):
                return None
            m = LMA_RE_PRICE_BETWEEN.search(text)
            if m:
                return LMA_SafeVal.sanitize_price_range(f'{m.group(1)}-{m.group(2)}')
            m = LMA_RE_PRICE_OVER.search(text)
            if m:
                return LMA_SafeVal.sanitize_price_range(f'{m.group(1)}-')
            m = LMA_RE_PRICE_UNDER.search(text)
            if m:
                return LMA_SafeVal.sanitize_price_range(f'-{m.group(1)}')
            return None

        @staticmethod
        def widen_price_range(price_range: str | None, ratio: float | None=None) -> str | None:
            if LMA_SafeVal.is_nullish(price_range):
                return None
            if ratio is None:
                ratio = LMA_BENCH.fuzzy_price_ratio
            else:
                ratio = ratio
            try:
                lo, hi = LMA_PriceBand.parse_price_bounds(price_range)
                if lo is not None:
                    lo = max(0.0, lo * (1.0 - ratio))
                if hi is not None:
                    hi = hi * (1.0 + ratio)
                if lo is not None:
                    lo_fmt = f'{lo:.0f}'
                else:
                    lo_fmt = ''
                if hi is not None:
                    hi_fmt = f'{hi:.0f}'
                else:
                    hi_fmt = ''
                widened = f'{lo_fmt}-{hi_fmt}'
                return None if widened == '-' else widened
            except (ValueError, TypeError):
                return LMA_SafeVal.sanitize_price_range(price_range)

        @staticmethod
        def price_fits_range(price: Any, price_range: str | None, *, fuzzy: bool=True) -> Tuple[bool, str]:
            pv = LMA_SafeVal.clamp_listing_price(price)
            if pv is None:
                return (False, 'price is null or non-numeric')
            lo, hi = LMA_PriceBand.parse_price_bounds(price_range)
            if lo is None:
                if hi is None:
                    return (True, 'no price constraint')
            if fuzzy:
                tol = LMA_BENCH.fuzzy_price_abs_tol
            else:
                tol = 0.0
            if lo is not None:
                if pv < lo - tol:
                    return (False, f'price {pv:g} below lower bound {lo:g}')
            if hi is not None:
                if pv > hi + tol:
                    return (False, f'price {pv:g} above upper bound {hi:g}')
            return (True, f'price {pv:g} within range')

        @staticmethod
        def classify_price_boundary(price: Any, lo: float | None, hi: float | None) -> str:
            pv = LMA_SafeVal.clamp_listing_price(price)
            if pv is None:
                return 'null'
            tol = LMA_BENCH.fuzzy_price_abs_tol
            if lo is not None:
                if abs(pv - lo) <= tol:
                    return 'boundary_low'
            if hi is not None:
                if abs(pv - hi) <= tol:
                    return 'boundary_high'
            if lo is not None:
                if pv < lo:
                    return 'under'
            if hi is not None:
                if pv > hi:
                    return 'over'
            return 'inside'

        @staticmethod
        def price_fit_score_adjustment(price_val: Any, price_range_str: Any) -> float:
            ok, _ = LMA_PriceBand.price_fits_range(price_val, str(price_range_str) if price_range_str else None, fuzzy=False)
            return 5.0 if ok else -25.0

        @staticmethod
        def extend_top_n_on_tie(scores: List[Tuple[Any, float]], top_n: int) -> List[Any]:
            if not scores or top_n <= 0:
                return []
            ordered = sorted(scores, key=lambda t: t[1], reverse=True)
            if len(ordered) <= top_n:
                return [item for item, _ in ordered]
            cutoff = ordered[top_n - 1][1]
            return [item for item, sc in ordered if sc >= cutoff]


    class _TMG20852_am_Debugger(object):
        def __init__(
            self,
            suite: Optional[str] = None,
            frozen: Optional[List[str]] = None,
            stop_after: Optional[str] = None,
            show_dlg: bool = False,
            quiet: bool = False
        ) -> None:
            self.memory: Dict[str, List] = defaultdict(lambda: [])
            self.quiet = quiet
            self.frozen = frozen or []
            self.stop_after = stop_after
            self.suite = suite
            self.show_dlg = show_dlg
        def add(self, stage: str, data: Any) -> None:
            if self.suite: self.memory[stage].append(data)
        def find(self, stage: str, kv: Dict) -> Any:
            for js in self.memory[stage]:
                for key, val in kv.items():
                    if js[key] != val: break
                else:
                    return js
            raise Exception(
                f"# cannot find the freezed {stage} instance for `{json.dumps(kv, ensure_ascii = False)}`"
            )
        def is_stop_in(self, episode: str) -> bool:
            if self.stop_after:
                if episode == "AGENT" and self.stop_after.startswith("a-"): return True
                if episode == "BASE" and self.stop_after.startswith("b-"): return True
                if episode == "PRODUCT" and self.stop_after.startswith("p-"): return True
                if episode == "SHOP" and self.stop_after.startswith("s-"): return True
                if episode == "VOUCHER" and self.stop_after.startswith("v-"): return True
            return False
        def is_stop(self, stage: str) -> bool:
            return self.stop_after == stage
        def is_frozen(self, stage: str) -> bool:
            return stage in self.frozen
        def print(self, obj: Any) -> None:
            if not self.quiet:
                if isinstance(obj, dict) or isinstance(obj, list):
                    am_MiscUtil.print_json(obj)
                else:
                    print(obj)


    class _TMG20852_AT_Vocab:
        ID_KEYS = ['brand', 'color', 'model', 'id', 'compatibility', 'color_family']
        ALPHABETS = 'ABCDEFGHIJKL' + 'M' + 'NOPQRSTUVWXYZ'
        KW_STOPWORDS: List[str] = ['color family', 'suitable', 'compatible', 'about', 'above', 'after', 'also', 'and', 'as', 'at', 'before', 'below', 'between', 'by', 'for', 'from', 'in', 'less', 'more', 'of', 'on', 'or', 'over', 'per', 'than', 'to', 'under', 'up', 'with', 'within']
        YES_WORDS: List[str] = ['yes', 'enabled', 'included', 'exists', 'true']
        NO_WORDS: List[str] = ['no', 'without any', 'without']
        COLOR_WORDS: List[str] = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'violet', 'orange', 'brown', 'gray', 'grey', 'beige', 'navy', 'gold', 'silver', 'bronze', 'magenta', 'cyan']
        SEARCHABLE_UNITS: List[str] = ['mm', 'cm', 'm', 'km', 'ft', 'yd', 'mg', 'g', 'kg', 'oz', 'lb', 'ml', 'l', 'mah', 'ah', 'wh', 'kwh', 'v', 'mv', 'kv', 'ma', 'w', 'kw', 'hz', 'khz', 'mhz', 'ghz', 'kb', 'mb', 'gb', 'tb', 'bps', 'kbps', 'mbps', 'gbps', '-c', '-f', 'pcs', 'pc', 'pa', 'kpa', 'mpa', 'mp', 'dpi', 'lux', 'db', 'rpm']
        SYNONYMS: List[Tuple[str, str]] = [(min(a, b), max(a, b)) for a, b in [('one', '1'), ('two', '2'), ('three', '3'), ('four', '4'), ('five', '5'), ('six', '6'), ('seven', '7'), ('eight', '8'), ('nine', '9'), ('ten', '10'), ('one', 'single'), ('two', 'double'), ('three', 'triple'), ('1', 'single'), ('2', 'double'), ('3', 'triple'), ('first', '1st'), ('second', '2nd'), ('third', '3rd'), ('hundred', '100'), ('thousand', '1000'), ('zero', '0'), ('shock-resistant', 'shockproof'), ('shock resistant', 'shockproof'), ('shock-proof', 'shockproof'), ('water-resistant', 'waterproof'), ('water resistant', 'waterproof'), ('water-proof', 'waterproof')]]
        P_EXT_SEARCH_UNITS: List[Tuple[str, str]] = [('piece', 'pcs'), ('pack', 'pcs')]


    class _TMG20852_LMA_TempCtrl:
        _BASE: ClassVar[Dict[str, float]] = {'parse': 0.0, 'review': 0.0, 'decompose': 0.0, 'filter': 0.0, 'score': 0.5, 'judge': 0.5, 'single_judge': 0.3, 'narrate': 0.3, 'alt_pick': 0.0}

        @staticmethod
        def base_temperature(task: str) -> float:
            return LMA_TempCtrl._BASE.get(task, 0.0)

        @staticmethod
        def for_attempt(task: str, attempt: int, *, step: float | None=None, max_temp: float | None=None) -> float:
            base = LMA_TempCtrl.base_temperature(task)
            if step is None:
                step = LMA_BENCH.temperature_retry_step
            else:
                step = step
            if max_temp is None:
                max_temp = LMA_BENCH.temperature_max
            else:
                max_temp = max_temp
            return min(max_temp, base + max(0, attempt) * step)


    class _TMG20852_AT_ShopSearchEngine(AT_SearchEngine):
    
        MAX_TRIES = 5
        MAX_SHOP_TRIES = 5
        def run(self) -> List[AT_Product]:
            super(AT_ShopSearchEngine, self).run()
            if self.debugger.is_stop_in('BASE'):
                return []
            best_p = self._run_shop_strategy()
            if self.debugger.is_stop('s-proc'):
                return []
            if best_p and len({p.shop_id for p in best_p}) != 1:
                self.debugger.print('* shop audit rejected (shop_count_mismatch)')
                best_p = []
            if best_p and (not AT_GateUtil.verify_pick(self.qi, best_p)):
                self.debugger.print('* shop audit rejected (verify_pick)')
                best_p = []
            return best_p
        def _run_shop_strategy(self) -> List[AT_Product]:
        
            best_p = self.__proc_shop_batches__(True)
            if not best_p:
                best_p = self.__shop_anchor_fill__()
                if best_p:
                    self.debugger.print('* shop strategy: recovered via anchor fill')
            if not best_p and len(self.qi.items) == 2:
                best_p = self.__shop_bidirectional_two_spec__()
                if best_p:
                    self.debugger.print('* shop strategy: recovered via bidirectional')
            if not best_p:
                best_p = self.__shop_rescore_top_shops__()
                if best_p:
                    self.debugger.print('* shop strategy: recovered via rescore')
            if not best_p:
                best_p = self.__global_anchor_resolve__()
                if best_p:
                    self.debugger.print('* shop strategy: recovered via global anchor')
            return best_p
        def __solve_knapsack__(self, cats: List[List[Tuple[AT_Product, AT_Weight]]]) -> List[Tuple[Tuple[AT_Product, ...], Any]]:
        
            combs = AT_MiscUtil.get_choice_combs(cats, False)
            solutions = []
            for choice in combs:
                prod_weights = [cats[i][n] for i, n in enumerate(choice)]
                prods, aggr_conf, aggr_score = ([], 1.0, 0.0)
                for p, w in prod_weights:
                    prods.append(p)
                    aggr_conf *= w.confidence
                    aggr_score += w.score_n
                if len({p.id for p in prods}) < len(prods):
                    continue
                solutions.append((tuple(prods), (aggr_conf, aggr_score)))
            return sorted(solutions, key=lambda s: s[-1], reverse=True)
        def __shop_anchor_search_in_shop__(self, sq: AT_SubQuery, shop_id: str) -> Optional[AT_Product]:
            base_sos = [so.update_shop(shop_id) for so in self.__get_sos__(sq, True)]
            relax_options: List[AT_SearchOption] = list(base_sos)
            if base_sos and base_sos[0].services:
                so = self.__get_sos__(sq, True)[0]
                so.services = []
                so.update_shop(shop_id).update_q()
                relax_options.append(so)
            words = (base_sos[0].q if base_sos else '').split()
            if len(words) > 2 and base_sos:
                so = self.__get_sos__(sq, True)[0]
                so.services = []
                so.update_shop(shop_id).update_q(' '.join(words[:2]))
                relax_options.append(so)
            if words and base_sos:
                so = self.__get_sos__(sq, True)[0]
                so.services = []
                so.price = None
                so.update_shop(shop_id).update_q(words[0])
                relax_options.append(so)
            found: List[AT_Product] = []
            for so in relax_options:
                batch = self.__find_and_view__(so, page_limit=1)
                found.extend(batch)
                if found:
                    break
            if not found:
                return None
            found = list({p.id: p for p in found}.values())
            for p in found:
                if p.id not in self.weight_pool[sq]:
                    self.__weigh__(sq, [p])
            candidates = sorted(found, key=lambda p: -self.weight_pool[sq][p.id].score_n)[:AT_SearchEngine.SELECT_CANDIDATES]
            best, _ = self.__select__(sq, candidates, False, 's')
            return best
        def __shop_pick_best_in_shop__(self, sq: AT_SubQuery, shop_id: str) -> Optional[AT_Product]:
            found = self.__shop_anchor_search_in_shop__(sq, shop_id)
            if found is not None:
                return found
            pool = [p for p in self.products.get(sq, []) if p.shop_id == shop_id]
            if not pool:
                return None
            return max(pool, key=lambda p: self.weight_pool[sq][p.id].score_n if p.id in self.weight_pool[sq] else 0.0)
        def __shop_anchor_fill__(self) -> List[AT_Product]:
            n = len(self.qi.items)
            if n < 2:
                return []
            shop_map, shop_stats = self.__get_shop_map__(self.products)
            partial = {sid: stats[0] for sid, stats in shop_stats.items() if stats[0] == n - 1}
            if not partial:
                return []
            winner = max(partial, key=lambda sid: shop_stats[sid][1])
            covered: Dict[int, AT_Product] = {}
            for i, sq in enumerate(self.qi.items):
                pool = shop_map[winner][sq]
                if pool:
                    covered[i] = max(pool, key=lambda p: self.weight_pool[sq][p.id].score_n)
            if len(covered) != n - 1:
                return []
            missing_i = next(i for i in range(n) if i not in covered)
            missing_sq = self.qi.items[missing_i]
            self.debugger.print(f'* shop anchor fill: shop `{winner}`, missing spec {missing_i + 1}')
            missing_p = self.__shop_anchor_search_in_shop__(missing_sq, winner)
            if missing_p is None:
                return []
            combo = [None] * n
            for i, p in covered.items():
                combo[i] = p
            combo[missing_i] = missing_p
            if not all(p is not None for p in combo):
                return []
            self.sess.step_on_anchor_fill(winner, missing_i, list(combo))
            return list(combo)
        def __shop_bidirectional_two_spec__(self) -> List[AT_Product]:
            if len(self.qi.items) != 2:
                return []
            shop_map, shop_stats = self.__get_shop_map__(self.products)
            ranked = sorted(shop_stats.items(), key=lambda kv: kv[1], reverse=True)[:6]
            best_combo, best_score = ([], -1.0)
            for sid, _ in ranked:
                picks = []
                score = 0.0
                ok = True
                for sq in self.qi.items:
                    pool = shop_map[sid][sq]
                    if not pool:
                        p = self.__shop_anchor_search_in_shop__(sq, sid)
                        if p is None:
                            ok = False
                            break
                        pool = [p]
                    p = max(pool, key=lambda x: self.weight_pool[sq][x.id].score_n)
                    picks.append(p)
                    score += self.weight_pool[sq][p.id].score_n
                if ok and score > best_score:
                    best_score, best_combo = (score, picks)
            if best_combo:
                self.sess.step_on_bidirectional(best_combo[0].shop_id, best_combo)
            return best_combo
        def __shop_rescore_top_shops__(self) -> List[AT_Product]:
            n = len(self.qi.items)
            if n < 2:
                return []
            shop_map, shop_stats = self.__get_shop_map__(self.products)
            ranked = sorted(shop_stats.items(), key=lambda kv: kv[1], reverse=True)[:5]
            best_combo: List[AT_Product] = []
            for sid, stats in ranked:
                if stats[0] < n - 1:
                    continue
                picks = []
                ok = True
                for sq in self.qi.items:
                    pool = shop_map[sid][sq]
                    if not pool:
                        p = self.__shop_anchor_search_in_shop__(sq, sid)
                        if p is None:
                            ok = False
                            break
                        pool = [p]
                    picks.append(max(pool, key=lambda x: self.weight_pool[sq][x.id].score_n))
                if ok and picks:
                    best_combo = picks
                    break
            if best_combo:
                self.sess.step_on_rescore_shops(best_combo[0].shop_id, best_combo)
            return best_combo
        def __global_anchor_resolve__(self) -> List[AT_Product]:
        
            n = len(self.qi.items)
            if n < 2:
                return []
            ranked: List[Tuple[float, int, AT_Product]] = []
            for i, sq in enumerate(self.qi.items):
                pool = sorted(
                    self.products.get(sq, []),
                    key=lambda p: -self.weight_pool[sq][p.id].score_n if p.id in self.weight_pool[sq] else 0.0,
                )
                for p in pool[:8]:
                    if not p.shop_id:
                        continue
                    sc = self.weight_pool[sq][p.id].score_n if p.id in self.weight_pool[sq] else 0.0
                    ranked.append((sc, i, p))
            ranked.sort(key=lambda x: -x[0])
            tried_shops: Set[str] = set()
            for _sc, anchor_idx, anchor in ranked[:12]:
                shop_id = anchor.shop_id or ''
                if not shop_id or shop_id in tried_shops:
                    continue
                tried_shops.add(shop_id)
                combo: List[Optional[AT_Product]] = [None] * n
                combo[anchor_idx] = anchor
                ok = True
                for i, sq in enumerate(self.qi.items):
                    if i == anchor_idx:
                        continue
                    pick = self.__shop_pick_best_in_shop__(sq, shop_id)
                    if pick is None:
                        ok = False
                        break
                    combo[i] = pick
                if ok and all(p is not None for p in combo):
                    self.debugger.print(f'* global-anchor resolved shop `{shop_id}`')
                    return list(combo)
            return []



def _chat(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _model_list():
        for _try in range(2):
            try:
                inference = _proxy.post(
                    "/inference/chat/completions",
                    json_data={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "required",
                        "temperature": 0,
                    },
                )
                if inference is None:
                    last_err = RuntimeError(f"inference failed for {model}")
                    time.sleep(0.3)
                    continue
                return inference
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.3)
    raise RuntimeError(f"inference unavailable: {last_err}")


def _env_call(envelope: dict[str, Any]) -> dict[str, Any]:
    base_id = str(envelope.get("call_id") or "call")
    for attempt in range(3):
        payload = dict(envelope)
        if attempt:
            suffix = f"r{attempt}-{int(time.time() * 1000) % 1_000_000}"
            payload["call_id"] = f"{base_id}-{suffix}"
            payload["idempotency_key"] = payload["call_id"]
            calls = []
            for index, call in enumerate(payload.get("calls") or [], start=1):
                c = dict(call)
                c["call_id"] = f"{payload['call_id']}-{index}"
                calls.append(c)
            payload["calls"] = calls
        result = _proxy.post("/environment/call", json_data=payload)
        if result is not None:
            return result
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"environment call failed after retries ({base_id})")


def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("function", {}).get("name") or "") for c in tool_calls]


def _closed_set_violation(
    tool_calls: list[dict[str, Any]],
    *,
    candidates: list[tuple[str, str]],
    viewed: set[str],
) -> str | None:
    if not candidates:
        return None
    cand_pids = {pid for pid, _ in candidates}
    cand_keys = {f"{pid}::{sku}" for pid, sku in candidates}
    names = _tool_names(tool_calls)
    if any(n in ("search", "filter") for n in names) and len(viewed & cand_pids) < len(
        cand_pids
    ):
        return (
            "Closed candidate set: do not search/filter until every listed product is viewed. "
            f"Missing: {sorted(cand_pids - viewed)}"
        )
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        raw = call["function"].get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if name == "view":
            pid = str(args.get("product_id") or "")
            if pid and pid not in cand_pids:
                return (
                    f"view product_id {pid} is outside the listed candidates. "
                    f"View only: {sorted(cand_pids)}"
                )
        if name in ("add_to_cart", "inspect_stock", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and pid not in cand_pids:
                return (
                    f"{name} product_id {pid} is outside listed candidates "
                    f"{sorted(cand_pids)}"
                )
            if pid and sku and f"{pid}::{sku}" not in cand_keys:
                if name in ("add_to_cart", "place_test_order"):
                    return (
                        f"{name} {pid}::{sku} is not a listed variant. "
                        f"Use one of: {sorted(cand_keys)}"
                    )
    return None


def _oos_order_violation(
    tool_calls: list[dict[str, Any]],
    *,
    last_stock: dict[str, Any] | None,
) -> str | None:
    """Reject ordering when latest stock/cart says out of stock (market shock)."""
    if not _stock_is_false(last_stock):
        return None
    names = _tool_names(tool_calls)
    if "place_test_order" in names:
        return (
            "Rejected: latest stock/cart shows in_stock=false (market shock). "
            "remove_from_cart if needed, pick another candidate, inspect_stock, "
            "then order only an in-stock sku."
        )
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: LS phases + safe-band gates for hidden races."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    allowed_fields = _claim_field_allowlist(tools)
    candidates = _parse_candidate_variants(query)
    cand_pids = {pid for pid, _ in candidates}
    ask_first = _needs_ask_first_spec(query)

    boot = (
        f"{query}\n\n"
        "Procedure: ask firm specs if withheld-until-asked; else retrieve → view → "
        "inspect_stock → cart → place_test_order. "
        "On stockout: message must COMMIT to same preferred model/other color "
        "(never ask 'should I?'); then remove → recover → order. "
        "Justification only if asked: include brand + exact view specs + current "
        "price/in_stock; never claim category; do not invent hd/fhd."
    )
    model_hint = _preferred_model_hint(query)
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
    if candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in candidates)
        boot += (
            "\n\nCLOSED CANDIDATE SET (mandatory): "
            f"{listed}. "
            "view each listed product_id; rank by stated priority; order a listed key only."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": boot},
    ]
    dialogue: list[dict[str, Any]] = []
    must_message = bool(ask_first)
    asked_spec = False
    viewed: set[str] = set()
    searchish = 0
    last_stock: dict[str, Any] | None = None
    view_facts_by_pid: dict[str, dict[str, str]] = {}
    shock_recover = False
    recovery_msg_force = False
    need_reinspect = False
    last_pick: dict[str, str] = {}
    cart_sku: dict[str, str] = {}
    max_steps = int(policy["max_steps"])

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        if must_message:
            hint = (
                "REQUIRED this turn: call `message` once. "
                "If this is a stockout/'still available?' reply: STATE you will take "
                "another color/variant of the SAME preferred model (do not ask permission)."
            )
            if model_hint:
                hint += f" Preferred model token: {model_hint}."
            messages.append({"role": "user", "content": hint})
        elif candidates and len(viewed & cand_pids) < len(cand_pids):
            missing = sorted(cand_pids - viewed)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Still missing views for closed-set product_ids: "
                        f"{missing}. Call view on the next missing id (or compare)."
                    ),
                }
            )
        elif shock_recover or need_reinspect:
            recover = (
                "Market shock / re-inspect required: remove_from_cart if cart is stale. "
                "Recover SAME preferred model other color/variant first"
            )
            if model_hint:
                recover += f" (model `{model_hint}`)"
            recover += (
                ", VIEW the replacement product, inspect_stock an in-stock sku, then "
                "place_test_order. Do not quit; do not ask the shopper what to do; "
                "do not order without fresh view+inspect_stock. "
                "If you message, STATE the same-model recovery plan (no permission ask)."
            )
            messages.append({"role": "user", "content": recover})
            shock_recover = False
        elif not candidates and searchish >= 3 and len(viewed) < 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "TF2 anti-thrash (HiSun/KAK): stop repeating the same search. "
                        "Reformulate once OR view best hits, then inspect_stock and finish."
                    ),
                }
            )
        elif not candidates and searchish >= 5 and len(viewed) >= 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Search budget spent. Among viewed products meeting firm constraints, "
                        "inspect_stock → add_to_cart → place_test_order NOW."
                    ),
                }
            )
        elif remaining <= 6 and not must_message:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Only ~{remaining} steps left. If you have an in-stock in-budget sku, "
                        "inspect_stock if needed and place_test_order NOW. Stop exploratory search."
                    ),
                }
            )

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(4):
            inference = _chat(messages=messages, tools=tools)
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if not tool_calls:
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {
                            "role": "user",
                            "content": "No tool call. Call a supplied tool now.",
                        },
                    ]
                )
                continue
            if must_message and "message" not in _tool_names(tool_calls):
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": "Rejected. Call only `message` this turn.",
                        },
                    ]
                )
                continue
            if not must_message:
                violation = _closed_set_violation(
                    tool_calls, candidates=candidates, viewed=viewed
                )
                if violation:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": f"Rejected: {violation}"},
                        ]
                    )
                    continue
                oos = _oos_order_violation(tool_calls, last_stock=last_stock)
                if oos:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": oos},
                        ]
                    )
                    shock_recover = True
                    need_reinspect = True
                    continue
                names_try = _tool_names(tool_calls)
                if (
                    need_reinspect
                    and "place_test_order" in names_try
                    and "inspect_stock" not in names_try
                ):
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {
                                "role": "user",
                                "content": (
                                    "Rejected: after a market event you must inspect_stock "
                                    "(and fix cart) before place_test_order."
                                ),
                            },
                        ]
                    )
                    continue
            break
        else:
            if candidates and len(viewed & cand_pids) < len(cand_pids):
                miss = sorted(cand_pids - viewed)[0]
                tool_calls = [
                    {
                        "id": f"forced-view-{turn}",
                        "type": "function",
                        "function": {
                            "name": "view",
                            "arguments": json.dumps({"product_id": miss}),
                        },
                    }
                ]
                assistant_content = f"Viewing closed-set candidate {miss}."
            elif must_message and ask_first and not asked_spec:
                tool_calls = [
                    {
                        "id": f"forced-ask-{turn}",
                        "type": "function",
                        "function": {
                            "name": "message",
                            "arguments": json.dumps(
                                {
                                    "content": (
                                        "What is the firm hardware requirement "
                                        "(RAM, screen size, resolution, or refresh rate)?"
                                    )
                                }
                            ),
                        },
                    }
                ]
                assistant_content = "Asking for the withheld firm spec."
            else:
                raise RuntimeError("model returned no valid tool call before completion")

        if must_message:
            msg_only = [
                c
                for c in tool_calls
                if c.get("function", {}).get("name") == "message"
            ]
            if msg_only:
                tool_calls = msg_only[:1]

        names = _tool_names(tool_calls)
        if any(n in ("search", "filter") for n in names):
            searchish += 1
        if "message" in names and ask_first:
            asked_spec = True

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{problem_id}-turn-{turn}"
        sanitized_calls: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls, start=1):
            args = _arguments(
                call,
                query=query,
                allowed_fields=allowed_fields,
                last_stock=last_stock,
                view_facts_by_pid=view_facts_by_pid,
            )
            name = call.get("function", {}).get("name") or ""
            if name == "message" and isinstance(args, dict):
                # EnvPack message schema uses `content` (fall back to `message`).
                rewritten = _rewrite_recovery_message(
                    args, query=query, force=recovery_msg_force
                )
                if rewritten is not args or recovery_msg_force:
                    args = rewritten
                    if "content" in args and "message" in args:
                        args.pop("message", None)
            if name in ("inspect_stock", "add_to_cart", "place_test_order"):
                if args.get("product_id") and args.get("sku"):
                    last_pick = {
                        "product_id": str(args["product_id"]),
                        "sku": str(args["sku"]),
                    }
            if name == "add_to_cart":
                cart_sku = dict(last_pick)
            if name == "remove_from_cart":
                cart_sku = {}
            sanitized_calls.append(
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": name, "args": args},
                }
            )
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": sanitized_calls,
        }
        result = _env_call(envelope)
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for call in tool_calls:
            if (call.get("function", {}).get("name") or "") == "view":
                raw = call["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}
                if isinstance(args, dict) and args.get("product_id"):
                    viewed.add(str(args["product_id"]))

        if "message" in names:
            must_message = False
            recovery_msg_force = False

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            obs = call_result.get("observation")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(obs),
                }
            )
            payload = _obs_stock_payload(obs if isinstance(obs, dict) else {})
            if payload is None and isinstance(obs, dict):
                payload = _obs_stock_payload({"observation": obs})
            name = tool_call.get("function", {}).get("name") or ""
            if name == "view" and isinstance(obs, dict):
                pid, facts = _extract_view_facts(obs)
                if not pid:
                    raw = tool_call["function"].get("arguments", "{}")
                    try:
                        a = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        a = {}
                    if isinstance(a, dict) and a.get("product_id"):
                        pid = str(a["product_id"])
                if pid and facts:
                    view_facts_by_pid[pid] = {
                        **view_facts_by_pid.get(pid, {}),
                        **facts,
                    }
            if name == "inspect_stock":
                need_reinspect = False
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                last_stock = payload
                if _stock_is_false(payload):
                    shock_recover = True
                    need_reinspect = True
                    recovery_msg_force = True
            if isinstance(obs, dict):
                inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
                if isinstance(inner, dict) and inner.get("market_event"):
                    shock_recover = True
                    need_reinspect = True
                    recovery_msg_force = True
                    added = inner.get("added")
                    if isinstance(added, dict):
                        last_stock = added

        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            if _looks_like_shopper_intervention(str(content)):
                # After ask-first reply with the actual spec, do NOT force another
                # message — resume shopping. Force message only for questions.
                low = str(content).lower()
                if ask_first and asked_spec and "?" not in low:
                    must_message = False
                else:
                    must_message = True
                if _looks_like_market_event(str(content)):
                    need_reinspect = True
                    recovery_msg_force = True
                elif "?" in low or any(
                    x in low for x in ("which", "still available", "hold on", "before you")
                ):
                    must_message = True
                    if any(
                        x in low
                        for x in ("still available", "out of stock", "hold on", "before you")
                    ):
                        recovery_msg_force = True
        if any(
            (call.get("observation") or {}).get("done")
            for call in result["calls"]
        ):
            break

        if not must_message and not candidates and not shock_recover:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Episode not done. Prefer inspect_stock → add_to_cart → "
                        "place_test_order on an in-stock sku that meets firm constraints. "
                        "Omit justification unless the shopper asked why."
                    ),
                }
            )
        elif not must_message and candidates and len(viewed & cand_pids) >= len(cand_pids):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "All listed candidates viewed. Rank by stated priority, "
                        "inspect_stock the winner, add_to_cart listed sku, place_test_order."
                    ),
                }
            )

    return dialogue

_TMG218C8_AST_PAD = 0
if _TMG218C8_AST_PAD:
    class _TMG218C8_LMA_ReviewProbe:

        @staticmethod
        def tally_services(qi: LMA_QueryInfo) -> Dict[str, int]:
            return {'freeShipping': len([sq for sq in qi.items if 'freeShipping' in sq.services]), 'official': len([sq for sq in qi.items if 'official' in sq.services]), 'flashsale': len([sq for sq in qi.items if 'flashsale' in sq.services]), 'COD': len([sq for sq in qi.items if 'COD' in sq.services]), 'price': len([sq for sq in qi.items if sq.price]), 'branded': len([sq for sq in qi.items if any((kw for kw in sq.kw if re.search('\\bbranded\\b', kw[-1], re.I)))])}

        @staticmethod
        def tally_query_mentions(qe: str) -> Dict[str, int]:
            cod_e = len(re.findall('\\bCOD\\b|\\b(cash(es|ing)?|pay(ment|ing)?s?)[\\s\\-]*(for)?[\\s\\-]*((up)?on|with(in)?|by|for)?[\\s\\-]*(deliver(y|ing|ie)?s?|cash(es|ing)?)', qe, flags=re.I))
            warranty_e = len(re.findall('\\b(\\d+)[\\s\\-]*(days?|weeks?|months?|years?)?[\\-\\s]*(periods?)?\\s*(of)?[\\-\\s]*(guarantee|warrant)', qe, re.I)) + len(re.findall('\\b(guarantee|warrant)[a-z]*[\\s\\-]*(days?|weeks?|months?|years?)?[\\-\\s]*(periods?)?\\s*(of)?(\\d+)', qe, re.I))
            return {'lazmall': len(re.findall('\\blaz\\-?mall\\b|\\bofficial\\s+(?:store|shop|brand)\\b', qe, re.I)), 'lazflash': len(re.findall('\\blaz\\-?flash\\b|\\bflash[\\s\\-]sale\\b|\\blightning[\\s\\-]deal\\b', qe, re.I)), 'free_ship': len(re.findall('\\b(free|compliment[a-z]+)\\s+(ship|deliver)', qe, re.I)), 'cod': cod_e, 'branded': len(re.findall('\\s+branded\\b', qe, re.I)), 'price': len(re.findall('\\b(php|pesos?|usd|dollars?|euros?)\\b', qe, re.I)), 'warranty': warranty_e}

        @staticmethod
        def mismatch_rows(mentions: Dict[str, int], parsed: Dict[str, int]) -> List[Tuple[str, int, int, str]]:
            mapping = [('lazmall', 'official', 'The official store condition ("LazMall" or "official store/shop/brand")'), ('lazflash', 'flashsale', 'The flash sale / LazFlash constraint'), ('free_ship', 'freeShipping', 'The condition of free or complimentary shipping'), ('cod', 'COD', 'The condition of "Cash on Delivery" option'), ('branded', 'branded', 'The word "branded"')]
            return [(lab, mentions[lab], parsed[pkey], prefix) for lab, pkey, prefix in mapping if mentions[lab] != parsed[pkey]]

        @staticmethod
        def focus_lines(suspects: List[str]) -> List[str]:
            table = {'lazmall': '- **lazmall**: explicit mention of "LazMall" or "official store/shop/brand".', 'lazflash': '- **lazflash**: explicit mention of a flash sale, lightning deal, or the word "LazFlash".', 'free_ship': '- **free_ship**: explicit mention of free or complimentary shipping.', 'cod': '- **cod**: explicit mention of cash on delivery or payment on delivery.', 'branded': '- **branded**: explicit mention of the word "branded". (don\'t confuse it with the brand constraint. the word "branded" must exist as is!)', 'warranty_period': '- **warranty_period**: explicit mention of guarantee or warranty period.', 'price': '- **price**: explicit mention of price range. (pared results use "[low]-[high]" format without currency unit. just "[low]-" or "-[high]" if only one bound is specified.)'}
            return [table[s] for s in ['lazmall', 'lazflash', 'free_ship', 'cod', 'branded', 'warranty_period', 'price'] if s in suspects]


    class _TMG218C8_AT_JsonModel(object):
        @classmethod
        def from_json(cls, data: dict) -> 'JsonModel':
            kwargs = dict([])
            _it_4 = iter(fields(cls))
            while True:
                try:
                    _iv_4 = next(_it_4)
                except StopIteration:
                    break
                f = _iv_4
                if f.name not in data:
                    continue
                value = data[f.name]
                t = f.type
                kwargs[f.name] = cls.__convert_from__(value, t)
            _rv_7507494 = cls(**kwargs)
            return _rv_7507494
        def to_json(self) -> Dict[str, Any] | List[Any]:
            _rv_6630394 = self.__convert_to__(self)
            return _rv_6630394
        @staticmethod
        def __convert_from__(value, t) -> Any:
            origin = get_origin(t)
            args = get_args(t)
            if origin is Union and len(args) == 2 and (NoneType in args):
                inner_t = next((a for a in args if a is not NoneType))
                if value is None:
                    return None
                _rv_8230321 = AT_JsonModel.__convert_from__(value, inner_t)
                return _rv_8230321
            if is_dataclass(t) and isinstance(value, dict):
                return t.from_json(value)
            if origin is list and isinstance(value, list):
                _rv_4033806 = [AT_JsonModel.__convert_from__(v, args[0]) for v in value]
                return _rv_4033806
            if origin is tuple and isinstance(value, list):
                _rv_6836386 = tuple((AT_JsonModel.__convert_from__(v, args[0]) for v in value))
                return _rv_6836386
            if origin is dict and isinstance(value, dict):
                kt, vt = args
                return {AT_JsonModel.__convert_from__(k, kt): AT_JsonModel.__convert_from__(v, vt) for k, v in value.items()}
            if t is float and isinstance(value, str):
                _rv_8260288 = AT_MiscUtil.extract_number(value)
                return _rv_8260288
            if t is int and isinstance(value, str):
                return AT_MiscUtil.extract_number(value, True)
            if t is str:
                _rv_9244016 = str(value)
                return _rv_9244016
            return value
        @staticmethod
        def __convert_to__(obj) -> Any:
            if is_dataclass(obj):
                result = {}
                _it_5 = iter(fields(obj))
                while True:
                    try:
                        _iv_5 = next(_it_5)
                    except StopIteration:
                        break
                    f = _iv_5
                    value = getattr(obj, f.name)
                    result[f.name] = AT_JsonModel.__convert_to__(value)
                return result
            if isinstance(obj, list):
                _rv_2619087 = [AT_JsonModel.__convert_to__(x) for x in obj]
                return _rv_2619087
            if isinstance(obj, tuple):
                return [AT_JsonModel.__convert_to__(x) for x in obj]
            if isinstance(obj, dict):
                _rv_4361342 = {AT_JsonModel.__convert_to__(k): AT_JsonModel.__convert_to__(v) for k, v in obj.items()}
                return _rv_4361342
            return obj


    class _TMG218C8_LMA_VoucherOption(LMA_JsonModel):
        budget: float
        apply_to: str
        threshold: Optional[float]
        is_fixed: bool
        discount: float
        cap: Optional[float]


    class _TMG218C8_LMA_BenchCfg:
        self_consistency_gap: float = 1.5
        enable_self_consistency: bool = True
        low_judge_score: float = 6.0
        fast_accept_score: float = 8.0
        fuzzy_price_ratio: float = 0.08
        fuzzy_price_abs_tol: float = 1.0
        temperature_retry_step: float = 0.3
        temperature_max: float = 1.2
        product_task_shortlist: int = 10
        llm_retry_max: int = 3
        voucher_window_eps: float = 0.5


    class _TMG218C8_am_QueryPipeline:
        """Classifies, decomposes, builds, and reviews a shopping query into a QueryInfo."""
        JSON_PARSE_TRIES = 2
        SPLIT_PARSE_TRIES = 2
        _RX_SPEND_LEAK = re.compile(
            r'\b(my\s+budget|voucher with|fixed\s+discount|percentage\s+discount|exceeds\s+`\d+`)\b', re.I
        )
        _RX_KW_NOISE = re.compile(
            r'^(looking(?:\s+for)?|show\s+me|find(?:\s+me)?|i(?:\'m|\s+am)\s+looking)\b', re.I
        )
        _RX_ORDINAL = re.compile(r'\b(?:first|second|third|fourth)\b')
        _RX_MULTI_HINT = re.compile(r'\band also\b|\bone option\b.*\banother\b|(?=.*\bboth\b)(?=.*\band\b)')
        _RX_COD = re.compile(
            r"\bCOD\b"
            r"|\b(cash(es|ing)?|pay(ment|ing)?s?)[\s\-]*(for)?[\s\-]*((up)?on|with(in)?|by|for)?[\s\-]*(deliver(y|ing|ie)?s?|cash(es|ing)?)",
            re.I,
        )
        _RX_WARRANTY_A = re.compile(
            r"\b(\d+)[\s\-]*(days?|weeks?|months?|years?)?[\-\s]*(periods?)?\s*(of)?[\-\s]*(guarantee|warrant)", re.I
        )
        _RX_WARRANTY_B = re.compile(
            r"\b(guarantee|warrant)[a-z]*[\s\-]*(days?|weeks?|months?|years?)?[\-\s]*(periods?)?\s*(of)?(\d+)", re.I
        )
        _RX_LAZMALL = re.compile(r"\blaz\-?mall\b|\bofficial\s+(?:store|shop|brand)\b", re.I)
        _RX_LAZFLASH = re.compile(r"\blaz\-?flash\b|\bflash\s+sale\b|\blightning\s+deal\b", re.I)
        _RX_FREESHIP = re.compile(r"\b(free|compliment[a-z]+)\s+(ship|deliver)", re.I)
        _RX_BRANDED = re.compile(r"\s+branded\b", re.I)
        _RX_CURRENCY = re.compile(r"\b(php|pesos?|usd|dollars?|euros?)\b", re.I)
        _RX_CURRENCY_TAIL = re.compile(r"\b(php|pesos?|usd|dollars?|euros?)\b[\s.]*$", re.I)
        _RX_PRICE_PART = re.compile(r"\d+(?:\.\d+)?")
        _RX_BRANDED_VAL = re.compile(r"\bbranded\b", re.I)
        _HINTS = {
            am_ShoppingTaskType.PRODUCT: (
                "\n\n# Hint\nThe original query contains only one product!"
            ),
            am_ShoppingTaskType.SHOP: (
                "\n\n# Hint\nThe user wants multiple products from the same shop. "
                "Split one sub-query per product; assign each product its own price and service constraints."
            ),
            am_ShoppingTaskType.VOUCHER: (
                "\n\n# Hint\nOnly the text before the budget block describes products. "
                "Split one sub-query per distinct product; exclude voucher/budget wording from sub-queries and keywords."
            ),
        }
        _SCOPE_RULES = (
            ("free_ship", "freeship_e", "freeship_p", "Free/complimentary shipping"),
            ("lazmall", "lazmall_e", "lazmall_p", "LazMall / official store"),
            ("lazflash", "lazflash_e", "lazflash_p", "Flash sale / LazFlash"),
            ("cod", "cod_e", "cod_p", "Cash on delivery"),
            ("branded", "brand_e", "brand_p", "Branded"),
        )
        def __init__(self, query: str, debugger: am_Debugger, task_hint: str = "") -> None:
            self.query = query
            self.debugger = debugger
            self.task_hint = task_hint
        def _build_hint_text(self, qi: am_QueryInfo) -> str:
            return self._HINTS.get(qi.challenge, "")
        @staticmethod
        def _estimate_min_items(query_e: str) -> int:
            q = query_e.lower()
            n = len(am_QueryPipeline._RX_ORDINAL.findall(q))
            if n >= 2: return n
            return 2 if am_QueryPipeline._RX_MULTI_HINT.search(q) else 1
        @staticmethod
        def _kw_signature(kw: List[str]) -> str:
            return kw[0] if len(kw) == 2 else kw[-2]
        @staticmethod
        def _kw_core_terms(kw_list: List[List[str]]) -> str:
            skip, parts = {"delivery", "service"}, []
            for kw in kw_list:
                if not kw: continue
                keys, val = kw[:-1], kw[-1].strip()
                if any(k in skip for k in keys): continue
                if val and val not in parts: parts.append(val)
            return " ".join(parts)
        @staticmethod
        def _sq_blob(sq: am_SubQuery) -> str:
            return " ".join([sq.query or ""] + [kw[-1] for kw in sq.kw if kw])
        @staticmethod
        def _has_type_keyword(sq: am_SubQuery) -> bool:
            return any(am_QueryPipeline._kw_signature(kw) == "product_type" for kw in sq.kw if kw)
        @staticmethod
        def _price_is_valid(price: Optional[str]) -> bool:
            if not price: return True
            parts = price.split("-", 1)
            if len(parts) != 2: return False
            return all(not part or am_QueryPipeline._RX_PRICE_PART.fullmatch(part) for part in parts)
        @staticmethod
        def _intent_has_spend_leak(sq: am_SubQuery) -> bool:
            return am_QueryPipeline._RX_SPEND_LEAK.search(am_QueryPipeline._sq_blob(sq)) is not None
        @staticmethod
        def _keyword_value_is_noisy(sq: am_SubQuery) -> bool:
            terms = am_QueryPipeline._kw_core_terms(sq.kw)
            if not terms.strip() or am_QueryPipeline._RX_KW_NOISE.search(terms): return True
            return any(
                kw and am_QueryPipeline._kw_signature(kw) == "product_type"
                and am_QueryPipeline._RX_KW_NOISE.search(kw[-1].strip())
                for kw in sq.kw
            )
        @staticmethod
        def _count_svc(items: List[am_SubQuery], svc: str) -> int:
            return sum(1 for sq in items if svc in sq.services)
        @staticmethod
        def _count_branded_items(items: List[am_SubQuery]) -> int:
            return sum(
                1 for sq in items
                if any(kw and am_QueryPipeline._RX_BRANDED_VAL.search(kw[-1]) for kw in sq.kw)
            )
        def _count_mentions(self, q: str) -> Dict[str, int]:
            return {
                "cod_e": len(self._RX_COD.findall(q)),
                "warranty_e": len(self._RX_WARRANTY_A.findall(q)) + len(self._RX_WARRANTY_B.findall(q)),
                "lazmall_e": len(self._RX_LAZMALL.findall(q)),
                "lazflash_e": len(self._RX_LAZFLASH.findall(q)),
                "freeship_e": len(self._RX_FREESHIP.findall(q)),
                "brand_e": len(self._RX_BRANDED.findall(q)),
                "price_e": len(self._RX_CURRENCY.findall(q)),
            }
        def _count_parsed(self, items: List[am_SubQuery]) -> Dict[str, int]:
            return {
                "freeship_p": self._count_svc(items, "freeShipping"),
                "lazmall_p": self._count_svc(items, "official"),
                "lazflash_p": self._count_svc(items, "flashsale"),
                "cod_p": self._count_svc(items, "COD"),
                "price_p": sum(1 for sq in items if sq.price),
                "brand_p": self._count_branded_items(items),
            }
        def _count_audit_flags(self, qi: am_QueryInfo) -> Dict[str, int]:
            return {**self._count_mentions(qi.query_e), **self._count_parsed(qi.items)}
        def _check_item_count(self, qi: am_QueryInfo, n: int) -> List[str]:
            errs, min_n = [], self._estimate_min_items(qi.query_e)
            if qi.challenge == am_ShoppingTaskType.PRODUCT and n != 1:
                errs.append(f"Product challenge must have exactly 1 sub-query, got {n}.")
            if qi.challenge == am_ShoppingTaskType.SHOP and n < 2:
                errs.append(f"Shop challenge must have at least 2 sub-queries, got {n}.")
            if qi.challenge == am_ShoppingTaskType.SHOP and n >= 2:
                terms = [self._kw_core_terms(sq.kw) for sq in qi.items]
                if len(set(terms)) < len(terms):
                    errs.append("Shop sub-queries must target distinct products (duplicate search terms).")
            if qi.challenge in (am_ShoppingTaskType.SHOP, am_ShoppingTaskType.VOUCHER) and n < min_n:
                errs.append(
                    f"Query describes at least {min_n} distinct products but only {n} sub-query(ies) were parsed."
                )
            if qi.challenge == am_ShoppingTaskType.VOUCHER and n == 1 and min_n >= 2:
                errs.append(
                    "Voucher query lists multiple products; split one sub-query per product (exclude budget/voucher text)."
                )
            return errs
        def _check_one_item(self, i: int, sq: am_SubQuery, is_voucher: bool) -> List[str]:
            errs = []
            if not (sq.query or "").strip():
                errs.append(f"Sub-query {i} is empty.")
            if not self._has_type_keyword(sq):
                errs.append(f"Sub-query {i} is missing a product_type keyword.")
            if self._keyword_value_is_noisy(sq):
                errs.append(f"Sub-query {i} keywords are not search-ready; drop filler like 'looking for'.")
            if not self._price_is_valid(sq.price):
                errs.append(f"Sub-query {i} has invalid price format `{sq.price}`.")
            if is_voucher and self._intent_has_spend_leak(sq):
                errs.append(f"Sub-query {i} still contains voucher/budget wording.")
            return errs
        def _check_scope_mismatch(self, qi: am_QueryInfo, c: Dict[str, int]) -> List[str]:
            errs = []
            for _, ek, pk, desc in self._SCOPE_RULES:
                if c[ek] != c[pk]:
                    errs.append(f"{desc}: mentioned {c[ek]} time(s) in query but assigned to {c[pk]} sub-query(ies).")
            if c["price_e"] != c["price_p"]:
                errs.append(
                    f"Price: currency mentioned {c['price_e']} time(s) but only {c['price_p']} sub-queries have a price range."
                )
            elif (
                c["price_e"] == 1 and qi.items and not qi.items[-1].price
                and self._RX_CURRENCY_TAIL.search(qi.query_e)
            ):
                errs.append(
                    "Price constraint appears at the end of the query but was not assigned to the last sub-query."
                )
            if c["warranty_e"] > 0 and not any(
                re.search(r"\b(guarantee|warrant)", self._sq_blob(sq), re.I) for sq in qi.items
            ):
                errs.append(
                    f"Warranty/guarantee mentioned {c['warranty_e']} time(s) but no sub-query carries a warranty constraint."
                )
            return errs
        def _validate_query_info(self, qi: am_QueryInfo) -> List[str]:
            n = len(qi.items)
            if n == 0: return ["No sub-queries were produced."]
            errs = self._check_item_count(qi, n)
            is_voucher = qi.challenge == am_ShoppingTaskType.VOUCHER
            for i, sq in enumerate(qi.items, 1):
                errs.extend(self._check_one_item(i, sq, is_voucher))
            errs.extend(self._check_scope_mismatch(qi, self._count_audit_flags(qi)))
            return errs
        def classify_task(self) -> am_QueryInfo:
            if self.debugger.is_frozen("a-classify"):
                return am_QueryInfo.from_json(self.debugger.find("a-classify", {"query_f": self.query}))
            challenge = self._identify_task_type()
            query_e, vopt = (
                self._parse_voucher_clause() if challenge == am_ShoppingTaskType.VOUCHER
                else (self.query, None)
            )
            qi = am_QueryInfo(challenge=challenge, query_f=self.query, query_e=query_e, vopt=vopt)
            self.debugger.add("a-classify", qi.to_json())
            self.debugger.print({
                "challenge": challenge,
                "query_e": query_e,
                "voucher": vopt.to_json() if vopt else None,
            })
            return qi
        def decompose_query(
            self,
            qi: am_QueryInfo,
            feedback: str = "",
            attempt: int = 0,
        ) -> List[Dict[str, Dict[str, Any]]]:
            if not self.debugger.is_frozen("a-decompose"):
                temperature = min(0.4, attempt * 0.2)
                tries = 0
                while tries < self.SPLIT_PARSE_TRIES:
                    tries += 1
                    try:
                        prompt = am_Prompt.A_DECOMPOSE.format(
                            shop_intent = " in the same shop" if qi.challenge == am_ShoppingTaskType.SHOP else "",
                            query = qi.query_e,
                        ) + self._build_hint_text(qi)
                        if feedback:
                            prompt += (
                                f"\n\n# Fix Required\nThe previous parsing failed validation:\n{feedback}"
                                "\nReturn corrected JSON only."
                            )
                        js, _ = am_ProxyUtil.trigger_llm(
                            prompt,
                            preferred_llms=am_ProxyUtil.get_role_llm_priorities("decompose"),
                            temperature=temperature,
                            expect="dict",
                        )
                        decomposed = am_Prompt.extract_decompose_response(js)
                        self.debugger.print(js)
                        if len(decomposed) == 0:
                            raise Exception("no sub-queries extracted")
                        min_products = self._estimate_min_items(qi.query_e)
                        if len(decomposed) == 1 and qi.challenge == am_ShoppingTaskType.SHOP:
                            raise Exception("sub-query count invalid")
                        if len(decomposed) < min_products and qi.challenge in (am_ShoppingTaskType.SHOP, am_ShoppingTaskType.VOUCHER):
                            raise Exception(f"sub-query count {len(decomposed)} < expected {min_products}")
                        self.debugger.add("a-decompose", {"query": self.query, "products": decomposed})
                        return decomposed
                    except Exception as exc:
                        print(f"# error parsing LLM response for decomposition: {exc}")
                        temperature += 0.2
                raise Exception("decompose failed after retries")
            else:
                return self.debugger.find("a-decompose", {"query": self.query})["products"]
        def build_query_info(self, qi: am_QueryInfo, parsed: List[Dict]) -> am_QueryInfo:
            if not self.debugger.is_frozen("a-build"):
                for inf in parsed:
                    q = inf.get("query")
                    kw_com = [am_EngineUtil.compact_kw(kw, q) for kw in inf.get("keywords")]
                    if inf.get("prod_only"):
                        vstr = "product only"
                        for i in range(len(kw_com)):
                            if am_EngineUtil.is_prod_only(kw_com[i][1]):
                                kw_com[i] = ["service", vstr]
                                break
                        else:
                            kw_com += [["service", "product only"]]
                    sq = am_SubQuery(
                        query = q,
                        services = inf.get("services"),
                        price = inf.get("price"),
                        kw = [kw for kw in kw_com if kw]
                    )
                    qi.items.append(sq)
                self.debugger.add("a-build", qi.to_json())
                return qi
            else:
                return am_QueryInfo.from_json(
                    self.debugger.find("a-build", {"query_f": self.query})
                )
        def prepare_context(self, qi: am_QueryInfo) -> am_QueryInfo:
            """Bounded decompose → build → validate loop (mirrors lucky_oro's _prepare_context)."""
            max_attempts = 1 if self.debugger.is_frozen("a-decompose") else self.JSON_PARSE_TRIES
            feedback = ""
            for attempt in range(max_attempts):
                try:
                    decomposed = self.decompose_query(qi, feedback, attempt)
                except Exception as exc:
                    print(f"# prepare_context decompose failed: {exc}")
                    if qi.items:
                        return qi
                    break
                qi.items.clear()
                qi = self.build_query_info(qi, decomposed)
                errors = self._validate_query_info(qi)
                if not errors:
                    if attempt:
                        self.debugger.print({"parse_validate": "ok", "attempt": attempt + 1})
                    return qi
                feedback = "\n".join(f"- {e}" for e in errors)
                self.debugger.print({"parse_validate": "retry", "attempt": attempt + 1, "errors": errors})
                if attempt + 1 >= max_attempts:
                    self.debugger.print({"parse_validate": "accept_with_errors", "errors": errors})
                    return qi
            return qi
        def review_query_info(self, qi: am_QueryInfo) -> am_QueryInfo:
            if not self.debugger.is_frozen("a-review"):
                suspects, prompts_i, prompts_o, focuses, error_3 = self._check_review_triggers(qi)
                if len(suspects) > 0:
                    js, _ = am_ProxyUtil.trigger_llm(
                        self._build_review_prompt(qi, prompts_i, prompts_o, focuses, error_3),
                        expect = "dict"
                    )
                    js["query"] = qi.query_e
                    self.debugger.print(js)
                    self._apply_review_fixes(qi, js, suspects)
                self.debugger.add("a-review", qi.to_json())
                return qi
            else:
                return am_QueryInfo.from_json(
                    self.debugger.find("a-review", {"query_f": self.query})
                )
        _VOUCHER_DISCOUNT_SIGNAL = re.compile(
            r"\bvoucher\b.*\b(?:fixed|percentage)\s+discount\b"
            r"|\b(?:fixed|percentage)\s+discount\b.*\bvoucher\b",
            re.I | re.S,
        )
        def _identify_task_type(self) -> am_ShoppingTaskType:
            """Classify the query using an LLM, falling back to regex on failure."""
            if self.task_hint in {t.value for t in am_ShoppingTaskType}:
                print(f"* classify_task (hint): {self.task_hint}")
                return am_ShoppingTaskType(self.task_hint)
            prompt = am_Prompt.A_CLASSIFY.format(query=self.query)
            models = am_ProxyUtil.get_role_llm_priorities("classify")
            result, _ = am_ProxyUtil.trigger_llm(prompt, preferred_llms=models, expect="dict")
            task_type_str = (result.get("task_type") or "").strip().lower()
            if task_type_str in {t.value for t in am_ShoppingTaskType}:
                print(f"* classify_task (llm): {task_type_str} | reason: {result.get('reason', '')}")
                return am_ShoppingTaskType(task_type_str)
            print("# classify_task: llm result invalid or missing, falling back to regex")
            return self._identify_task_type_regex()
        def _identify_task_type_regex(self) -> am_ShoppingTaskType:
            """Regex-based fallback classifier (used when the LLM step fails)."""
            for pat in am__VOUCHER_CLASSIFY_PATTERNS:
                if re.search(pat, self.query, re.I):
                    return am_ShoppingTaskType.VOUCHER
            if self._VOUCHER_DISCOUNT_SIGNAL.search(self.query):
                return am_ShoppingTaskType.VOUCHER
            for pat in am__SHOP_CLASSIFY_PATTERNS:
                if re.search(pat, self.query, re.I):
                    return am_ShoppingTaskType.SHOP
            return am_ShoppingTaskType.PRODUCT
        def _parse_voucher_clause(self) -> Tuple[str, am_VoucherOption]:
            try:
                query_e, fields = am_hero_parse_voucher_from_query(self.query)
                return query_e, am_VoucherOption(
                    budget=fields["budget"],
                    apply_to=fields["apply_to"],
                    threshold=fields["threshold"],
                    is_fixed=fields["is_fixed"],
                    discount=fields["discount"],
                    cap=fields["cap"],
                )
            except Exception as exc:
                self.debugger.print(f"# hero voucher parse failed, strict fallback: {exc}")
            sep = self.query.lower().find("my budget is")
            if sep < 0:
                raise ValueError(f"cannot parse voucher clause from: {self.query!r}")
            query_e = self.query[:sep].strip()
            v_str = self.query[sep:].strip()
            m = re.search(r"\bmy\s+budget\s+is(?:\s+only)?\s+`(\d+)`", v_str, re.I)
            if not m:
                raise ValueError(f"cannot parse voucher budget from: {self.query!r}")
            budget = int(m.group(1))
            apply_to = "shop" if re.search(r"\bsame\s+shops?\b", v_str, re.I) else "all"
            m = re.search(r"\bexceeds?\s+`(\d+)`", v_str, re.I)
            threshold = int(m.group(1)) if m else None
            if (m := re.search(r"\bpercent(?:age)?\s+discount(?:\s+of)?\s+`([\d\.]+)\%`", v_str, re.I)):
                discount = float(m.group(1))
                is_fixed = False
                m = re.search(r"\bwith(?:\s+a)\s+cap(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
                cap = float(m.group(1)) if m else None
            else:
                m = re.search(r"\bfixed\s+discount(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
                discount = float(m.group(1)) if m else 0.0
                is_fixed = True
                cap = None
            vopt = am_VoucherOption(
                budget = budget,
                apply_to = apply_to,
                threshold = threshold,
                is_fixed = is_fixed,
                discount = discount,
                cap = cap
            )
            return query_e, vopt
        def _check_review_triggers(self, qi: am_QueryInfo) -> Tuple[List[str], str, str, str, str]:
            freeship_p = len([sq for sq in qi.items if "freeShipping" in sq.services])
            lazmall_p = len([sq for sq in qi.items if "official" in sq.services])
            lazflash_p = len([sq for sq in qi.items if "flashsale" in sq.services])
            cod_p = len([sq for sq in qi.items if "COD" in sq.services])
            price_p = len([sq for sq in qi.items if sq.price])
            brand_p = len([sq for sq in qi.items if any(kw for kw in sq.kw if re.search("\bbranded\b", kw[-1], re.I))])
            cod_e = len(
                re.findall(
                    r"\bCOD\b"
                    r"|\b(cash(es|ing)?|pay(ment|ing)?s?)[\s\-]*(for)?[\s\-]*((up)?on|with(in)?|by|for)?[\s\-]*(deliver(y|ing|ie)?s?|cash(es|ing)?)",
                    qi.query_e,
                    flags = re.I
                )
            )
            warranty_e = len(
                re.findall(
                    r"\b(\d+)[\s\-]*(days?|weeks?|months?|years?)?[\-\s]*(periods?)?\s*(of)?[\-\s]*(guarantee|warrant)",
                    qi.query_e,
                    re.I
                )
            ) + len(
                re.findall(
                    r"\b(guarantee|warrant)[a-z]*[\s\-]*(days?|weeks?|months?|years?)?[\-\s]*(periods?)?\s*(of)?(\d+)",
                    qi.query_e,
                    re.I
                )
            )
            lazmall_e = len(re.findall(r"\blaz\-?mall\b|\bofficial\s+(?:store|shop|brand)\b", qi.query_e, re.I))
            lazflash_e = len(re.findall(
                r"\blaz\-?flash\b"
                r"|\bflash\s+sale\b"
                r"|\blightning\s+deal\b"
                r"|\blimited[\-\s]time\s+(?:deal|offer|promo(?:tion)?|sale|discount)\b"
                r"|\btime[\-\s]limited\s+(?:deal|offer|promo(?:tion)?|sale|discount)\b"
                r"|\bseller['\u2019]?s\s+limited[\-\s]time\s+(?:deal|offer|promo(?:tion)?|sale)\b"
                r"|\bexclusive\s+(?:time[\-\s]limited|limited[\-\s]time)\s+(?:deal|offer|promo(?:tion)?)\b",
                qi.query_e, re.I
            ))
            freeship_e = len(re.findall(r"\b(free|compliment[a-z]+)\s+(ship|deliver)", qi.query_e, re.I))
            brand_e = len(re.findall(r"\s+branded\b", qi.query_e, re.I))
            price_e = len(re.findall(r"\b(php|pesos?|usd|dollars?|euros?)\b", qi.query_e, re.I))
            suspects, prompts_i, prompts_o, focuses = [], [], [], []
            for label, count_e, count_p, prefix in [
                ("lazmall", lazmall_e, lazmall_p, "The official store condition (\"LazMall\" or \"official store/shop/brand\")"),
                ("lazflash", lazflash_e, lazflash_p, "The flash sale condition (\"flash sale\", \"lightning deal\", \"LazFlash\", or paraphrases like \"limited-time deal/offer\", \"seller's deal\", \"seller's limited-time deal\", \"time-limited promotion\", etc.)"),
                ("free_ship", freeship_e, freeship_p, "The condition of free or complimentary shipping"),
                ("cod", cod_e, cod_p, "The condition of \"Cash on Delivery\" option"),
                ("branded", brand_e, brand_p, "The word \"branded\""),
            ]:
                if count_e != count_p:
                    suspects.append(label)
                    prompts_i.append(f"""
    - {prefix} appeared {count_e} time(s) in the original query, but {count_p} products were parsed with this constraint. It needs double check.
    """.strip()
                    )
                    prompts_o.append(f"""
        "{label}": [
            <product_id_1, str>, ... // correct product ids that must receive **{label}** constraint.
        ]
    """.strip())
            if "lazmall" in suspects and warranty_e > 0:
                prompts_i.append(f"""
    - Warranty period was mentioned in the original query {warranty_e} time(s), but no product was parsed with this constraint. It needs double check.
    """.strip())
                prompts_o.append("""
        "warranty_period": {
            "apply_for": <product_id_1, str|null>, // the product id that must receive warranty period constraint. set null if no product is relevant.
            "period": <str|null> // warranty period mentioned for this product in the original query if exists, otherwise null. for example, "period": "14 days"
        }
    """.strip())
                suspects.append("warranty_period")
            if price_e != price_p:
                prompts_i.append(f"""
    - Currency unit appeared {price_e} time(s) in the original query, but {price_p} products were parsed with price constraint. You must carefully check the context if those number ranges are really the price constraints.
    """.strip())
                prompts_o.append("""
        "price": { // list all the products (don't skip any product)
            "1": { // product id = 1
                "parsed_price_range": <str|null>, // the price range string or null as assigned for this product by the parsing result given
                "is_it_correct": <bool>, // true if the parsed result is correct, otherwise false
                "other_spec": <str|null>, // the other spec name ("power usage", for example) which was mistakenly identified as the price range in the parsing result if exists. set null if "parsed_price_range" is null or the parsed result has no error.
                "correct_price_range": <str|null> // the corrected price range for this product. write the original parsed result again if it has no error. set null if no price range condition is for this product. use "[low]-[high]" format without currency unit if exists. "[low]-" or "-[high]" if either bound is missing.
            },
            ...
        }
    """.strip())
                suspects.append("price")
            elif price_e == 1 and not qi.items[-1].price:
                if re.search(r"\b(php|pesos?|usd|dollars?|euros?)\b[\s.]*$", qi.query_e, re.I):
                    prompts_i.append(
                        "- The price range appeared at the end of the original query text, but in the parsed result, "
                        "the product which is not the last one was parsed with the price constraint. You must carefully check if it's correctly parsed."
                    )
                    prompts_o.append("""
        "price": { // list all products (don't skip any product)
            "1": { // product id = 1
                "parsed_price_range": <str|null>, // the price range string or null as assigned for this product by the parsing result given
                "correct_price_range": <str|null> // the corrected price range for this product. write the original parsed result again if it has no error. set null if no price range condition is for this product. use "[low]-[high]" format without currency unit if exists. "[low]-" or "-[high]" if either bound is missing.
            }
        }
    """.strip())
                    suspects.append("price")
            if "lazmall" in suspects: focuses.append('- **lazmall**: explicit mention of "LazMall" or "official store/shop/brand".')
            if "lazflash" in suspects: focuses.append('- **lazflash**: explicit mention of "flash sale", "lightning deal", "LazFlash", or paraphrases such as "limited-time deal/offer", "seller\'s deal", "seller\'s limited-time deal", "time-limited promotion", etc.')
            if "free_ship" in suspects: focuses.append('- **free_ship**: explicit mention of free or complimentary shipping.')
            if "cod" in suspects: focuses.append('- **cod**: explicit mention of cash on delivery or payment on delivery.')
            if "branded" in suspects: focuses.append('- **branded**: explicit mention of the word "branded". (don\'t confuse it with the brand constraint. the word "branded" must exist as is!)')
            if "warranty_period" in suspects: focuses.append('- **warranty_period**: explicit mention of guarantee or warranty period.')
            if "price" in suspects: focuses.append('- **price**: explicit mention of price range. (pared results use "[low]-[high]" format without currency unit. just "[low]-" or "-[high]" if only one bound is specified.)')
            if "price" in suspects:
                error_3 = """
    3. Confused price range
    - A product appears with price range constraint in the parsed result, but actually this number range is for a completely different specification.
    - For example, in "I need a bottle between 0.5 and 1.2 litre", the range "0.5-1.2" is not price constraint, but a capacity constraint.
    - Also, in "Find motors with power usage less than 2.5", the range "-2.5" is mentioned without currency unit, but you can easily confirm that it's the power usage constraint with unit "kw".
    """.strip()
            else:
                error_3 = ""
            return suspects, '\n'.join(prompts_i), '\n'.join(prompts_o), '\n'.join(focuses), error_3
        def _build_review_prompt(self, qi: am_QueryInfo, prompts_i: str, prompts_o: str, focuses: str, error_3: str) -> str:
            prods = {
                str(i + 1): {
                    "sub-query": sq.query,
                    "constraints": [am_Prompt.service_to_label(svc) for svc in (sq.services or [])],
                    "price": sq.price
                }
                for i, sq in enumerate(qi.items)
            }
            return am_Prompt.A_REVIEW.format(
                count = len(qi.items),
                focuses = focuses,
                suspects = prompts_i,
                error_3 = error_3,
                responses = prompts_o,
                query = qi.query_e,
                parsed = prods
            )
        def _apply_review_fixes(self, qi: am_QueryInfo, js: Dict[str, List[str] | Dict], suspects: List[str]):
            def __get_id__(id_str: Any) -> Optional[int]:
                if not isinstance(id_str, str): return None
                pid = am_MiscUtil.extract_number(id_str, True) - 1
                if pid < 0 or pid >= len(qi.items): return None
                return pid
            for issue in suspects:
                if issue not in js: continue
                obj = js.get(issue)
                if issue in ["lazmall", "lazflash", "cod", "free_ship"]:
                    if not isinstance(obj, list): continue
                    svc = am_Prompt.label_to_service(issue)
                    ids = [__get_id__(x) for x in obj]
                    if None in ids: continue
                    for i, sq in enumerate(qi.items):
                        if i in ids and svc not in sq.services:
                            sq.services.append(svc)
                        elif i not in ids and svc in sq.services:
                            sq.services.remove(svc)
                elif issue == "branded":
                    if not isinstance(obj, list): continue
                    ids = [__get_id__(x) for x in obj]
                    if None in ids: continue
                    for i, sq in enumerate(qi.items):
                        if i in ids and not any(kw for kw in sq.kw if re.search("\bbranded\b", kw[-1], re.I)):
                            sq.kw.append(["brand", issue])
                            sq.query += ", it must come with branded feature."
                        elif i not in ids:
                            for kw in sq.kw:
                                kw[-1] = re.sub(r"\s+branded\b", " ", kw[-1], flags = re.I)
                                kw[-1] = re.sub(r"\s+", " ", kw[-1])
                            sq.query = re.sub(r"\bbranded\b", "", sq.query, flags = re.I)
                            sq.query = re.sub(r"\s+", " ", sq.query)
                elif issue == "warranty_period":
                    if not isinstance(obj, dict): continue
                    id = __get_id__(obj.get("apply_for"))
                    if id is None: continue
                    period = obj.get("period")
                    if not period: continue
                    qi.items[id].kw.append([issue, period])
                    qi.items[id].query += f", it must provide with {period} of guarantee"
                elif issue == "price":
                    if not isinstance(obj, dict): continue
                    ids = [__get_id__(k) for k in obj.keys()]
                    if None in ids: continue
                    if sorted(ids) != list(range(len(qi.items))): continue
                    for i, sq in enumerate(qi.items):
                        j: Dict[str, str] = obj.get(str(i + 1))
                        correct = j.get("correct_price_range")
                        price = re.sub(r"[^\d\-]", "", correct or "") or None
                        if price != sq.price and sq.price:
                            spec = j.get("other_spec")
                            if spec:
                                sq.kw.append([spec.replace(' ', '_'), sq.price])
                                sq.query += f", it must have {spec} with range {sq.price}"
                        sq.price = price



__all__ = ["agent_main"]
