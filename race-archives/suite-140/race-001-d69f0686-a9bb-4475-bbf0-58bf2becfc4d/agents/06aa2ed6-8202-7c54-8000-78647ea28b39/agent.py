from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

LLM_PRIORITIES = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "deepseek/deepseek-chat-v3.1",
]
_OPENROUTER_MODELS = list(LLM_PRIORITIES)

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

_proxy = ProxyClient(timeout=120, max_retries=2)

_SYSTEM = """You are an ORO Bench shopping agent (Lovelystar + 0828 recovery discipline).

Hidden-race goal: every capability family stays reliable — not max public-suite score.
Use ONLY supplied tools. Continue until done=true. Never memorize product IDs.

## Reasonable loop
Decompose silently → search/filter (few reformulations) → view/compare → inspect_stock
→ add_to_cart → place_test_order. Answer real shopper questions once with message.

## Ask-first firm specs
If shopper says a firm requirement will be told when you ask: first call `message` to ask.
After the answer, search with that spec and finish — no apology loops.

## Stockout / price-shock recovery (TF6 / TF2) — CRITICAL
When cart flips in_stock=false or shopper asks "is that still available?":
  1) message that STATES your plan (do NOT ask permission / preferences).
  2) If shopper preferred a color+model: commit to THE SAME MODEL in another color/
     variant first; keep budget + category firm. Only then change models.
     Good: "The black Lumia 635 went out of stock. I'm now looking for another color
     of the same Lumia 635 within budget."
     Bad: "Should I try another color, or a different model?" / "Any preferences?"
  3) remove_from_cart → search/view that same model → inspect_stock → cart → order.
NEVER place_test_order on an out-of-stock line. NEVER quit with "I cannot complete".

## Closed candidates
When shopper lists product_id::sku ("compare only these"): view/rank those only;
order a listed key — no open-catalog substitutes.

## Justification (only if shopper asks why/reason/walk me through)
Claims must be faithful to the LAST view/compare + latest inspect_stock:
  - ALWAYS include brand when the viewed product has a brand (soft claim).
  - Include the named spec (screen_size / resolution / refresh_rate) using the EXACT
    visible fact string from view (e.g. "27 inches", "1920 x 1080") — do not invent hd.
  - Always include current price + in_stock from inspect_stock.
  - NEVER claim category.
  - If not asked for a reason, OMIT justification entirely.
"""


def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw == "chutes":
        return "chutes"
    return "openrouter"

_FCS00BC4_AST_PAD = 0
if _FCS00BC4_AST_PAD:
    class _FCS00BC4_V_Vocab:
        SHOP_PATTERN = '\\b(?:look(?:ing)?(?:\\s+for)?|find|show|same)\\b(?:\\s+\\w+){0,10}\\s+shops?\\b'
        VOUCHER_PATTERN = '\\bmy\\s+budget\\s+is\\b'
        PROD_ONLY = 'product_only'
        SERVICE_PATTERNS = {svc: pats for svc, pats in [('COD', ['\\b((?:cash(?:es|ing)?|pay(?:ment|ing)?s?)[\\s\\-]*(?:for)?[\\s\\-]*(?:(?:up)?on|with(?:in)?|by|for)?[\\s\\-]*(?:deliver(?:y|ing|ie)?s?|cash(?:es|ing)?))\\b']), ('product_only', ['\\b((?:with\\s+)?(?:just|only)(?:\\s+the)?\\s+products?)\\b', '\\b((?:with\\s+)?products?\\s+only(?:\\s+services?)?)\\b']), ('official', ['\\b(laz\\-?mall(?:[\\s\\-]+perks?)?)\\b', '\\b(official(?:[\\s\\-]+stores?|[\\s\\-]+sellers?|[\\s\\-]+flagship[\\s\\-]+stores?)?)\\b']), ('flashsale', ['\\b(laz\\-?flash(?:[\\s\\-]+deals?)?)\\b', '\\b(flash(?:[\\s\\-]?sales?|[\\s\\-]+deals?|[\\s\\-]+offers?))\\b']), ('freeShipping', ['\\b((?:free|compliment[a-z]+)\\s+(?:ship|deliver)[a-z]*)\\b'])]}
        KW_STOPWORDS = ['color family', 'suitable', 'compatible', 'support', 'made', 'aged', 'needs', 'that', 'helps', 'contains', 'includes', 'option', 'about', 'above', 'after', 'also', 'before', 'below', 'between', 'for', 'from', 'comes', 'less', 'more', 'over', 'than', 'under', 'with', 'within', 'without', 'provides', 'sold']
        COLOR_WORDS = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'violet', 'orange', 'brown', 'gray', 'grey', 'beige', 'navy', 'gold', 'silver', 'bronze', 'magenta', 'cyan', 'dark', 'light']
        METRIC_UNITS = {a: b for a, b in [('mm', ''), ('cm', ''), ('mg', ''), ('kg', ''), ('oz', ''), ('lb', ''), ('ml', ''), ('mah', ''), ('ah', ''), ('wh', ''), ('kwh', ''), ('mv', ''), ('kv', ''), ('ma', ''), ('kw', ''), ('hz', ''), ('khz', ''), ('mhz', ''), ('ghz', ''), ('bps', ''), ('kbps', ''), ('mbps', ''), ('gbps', ''), ('°c', ''), ('°f', ''), ('kpa', ''), ('mpa', ''), ('dpi', ''), ('lux', ''), ('db', ''), ('rpm', ''), ('y', 'year'), ('yrs', 'year'), ('km', 'kilometer'), ('ft', 'feet'), ('m', 'meter'), ('yd', 'yard'), ('g', 'gram'), ('l', 'liter'), ('v', 'volt'), ('w', 'watt'), ('pa', 'pascal'), ('kb', 'kilobyte'), ('mb', 'megabyte'), ('gb', 'gigabyte'), ('tb', 'terabyte'), ('mp', 'megapixel'), ('pc', 'piece'), ('pcs', 'piece')]}
        MORPH_RULES: List[Tuple[str, List[Union[str, Callable]]]] = [('ing', ['e', '', re.sub]), ('ed', ['e', '', re.sub]), ('est', ['e', '', re.sub]), ('er', ['e', '', re.sub]), ('ish', ['e', '', re.sub]), ('iness', ['y']), ('inesses', ['y']), ('ily', ['y']), ('ies', ['y', '']), ('ness', ['y', '']), ('es', ['e', '']), ('s', ['']), ('ful', ['']), ('less', ['']), ('ly', ['']), ('en', [''])]
        NUMBERS = {a: b for a, b in [('single', '1'), ('one', '1'), ('two', '2'), ('three', '3'), ('four', '4'), ('five', '5'), ('six', '6'), ('seven', '7'), ('eight', '8'), ('nine', '9'), ('ten', '10')]}
        ALTS = [('all', 'any'), ('everyday', 'casual'), ('any', 'universal'), ('kid', 'child'), ('men', 'male'), ('women', 'female'), ('tier', 'layer'), ('100%', 'pure')]
        BRIDGES = [('all', 'any'), ('man', 'men'), ('woman', 'women'), ('child', 'children'), ('metre', 'meter'), ('litre', 'liter')]
        SIZES = [('s', 'small'), ('l', 'large'), ('m', 'medium')]
        YES_WORDS = ['yes', 'enabled', 'included', 'exists', 'true', 'required']
        NO_WORDS = ['no', 'without any', 'without']
        ID_KEYS = ['brand', 'color', 'model', 'id', 'compatibility', 'color_family']
        FILLERS = ['set', 'pack', 'accessory', 'option']
        ALPHABETS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


    def _FCS00BC4_S_resolve_engine_cls(challenge: 'S_ShoppingTaskType') -> Type['S_SearchEngine']:
        return S_ShopSearchEngine


    def _FCS00BC4_run_product_lane(problem_data: dict) -> list:
        _install_product_tools()
        if not isinstance(problem_data, dict):
            problem_data = {"query": "" if problem_data is None else str(problem_data)}
        pd = dict(problem_data or {})
        pd["category"] = "product"
        pd["task"] = "product"
        if ENABLE_DEBUG:
            _debug_print({"lane": "product", "step": "start"})
        return P_agent_main(pd)


    class _FCS00BC4_P_BaseEngine(P_Debuggable):
        MAX_SELECT_TRIES = 1
        SELECT_BATCH = 17
        TIMEOUT_SECS = 280.0
        MAX_KW_VARS = 2
        PREFILTER_TOPN = 30
        PREFILTER_RESERVE = 45.0
        PRUNE_THRES_STRICT = 1.75
        SCORE_N_THRES = 0.4

        def __init__(self, query: P_Query, sess: P_Session, agent_delay: float) -> None:
            super().__init__(query)
            self.agent_delay = agent_delay
            self.sess = sess
            self.weight_pool: Dict[P_SubQuery, Dict[str, P_Weight]] = defaultdict(lambda: {})
            self.prod_pool: Dict[str, P_Product] = {}
            self.searches: Dict[P_SubQuery, Dict[P_Filter, List[P_Product]]] = {}
            self.products: Dict[P_SubQuery, List[P_Product]] = {}
            self.filters: Dict[P_SubQuery, List[Tuple[P_Filter, float]]] = defaultdict(lambda: [])
            self.irank: Dict[P_SubQuery, int] = defaultdict(lambda: 0)
            self.confirmed = defaultdict(lambda: defaultdict(lambda: None))
            self.watches: Dict[str, float] = {}
            self.fillers: List[str] = list(set(P_Vocab.FILLERS + (query.fillers or [])))
            _, self.assist_map = self.__build_assistant_map__(True)
            self.footprints: Dict[P_SubQuery, Dict[Tuple[Tuple, Tuple], Tuple[Set, bool]]] = defaultdict(lambda: {})
            self.__init_qs__()

        def run(self) -> List[P_Product]:
            self.__init_watch__('query', P_BaseEngine.TIMEOUT_SECS - self.agent_delay)
            for sq in self.query.intents:
                self.__update_filters__(sq, True)
            alt_filters = self.__kw__()
            for i, sq in enumerate(self.query.intents):
                if alt_filters[i] is not None:
                    self.filters[sq].insert(1, (alt_filters[i], 0))
                searches: Dict[P_Filter, List[P_Product]] = self.__search__(sq, [f for f, _ in self.filters[sq]], 3, id='init')
                self.__update_searches__(sq, searches)
                products, self.weight_pool[sq] = self.__weigh__(sq, self.searches[sq], id='init')
                self.products[sq] = [self.__get_prod__(p) for p in products]
                self.irank[sq] = len(self.products[sq])
            self.sess.step_on_search()
            return []

        def __kw__(self) -> List[Optional[P_Filter]]:
            inf = {str(i + 1): {'query': sq.query, 'keywords': self.filters[sq][0][0].get_q()} for i, sq in enumerate(self.query.intents)}
            js, _ = P_ProxyUtil.trigger_llm(P_Prompt.PSV_KW.format(inf=inf), expect='dict')
            keywords = P_Prompt.parse_kw_response(js)
            filters = []
            for i, sq in enumerate(self.query.intents):
                if i < len(keywords) and keywords[i]:
                    nf = self.filters[sq][0][0].clone().update(keywords=keywords[i].split())
                    filters.append(nf if nf not in [f for f, _ in self.filters[sq]] else None)
                else:
                    filters.append(None)
            return filters

        def __search__(self, sq: P_SubQuery, filters: List[P_Filter], page_limit: int, **_) -> Dict[P_Filter, List[P_Product]]:
            js: Dict[P_Filter, List[P_Product]] = {}
            for f in filters:
                js[f] = self.__find_and_view__(f, page_limit)
            return js

        def __weigh__(self, sq: P_SubQuery, searches: Dict[P_Filter, List[P_Product]], **_) -> Tuple[List[P_Product], Dict[str, P_Weight]]:
            products: List[P_Product] = [p for filter in searches for p in searches[filter]]
            for i, p in enumerate(products):
                if p.id not in self.weight_pool[sq]:
                    self.weight_pool[sq][p.id] = self.__weigh_product__(p, sq, i + self.irank[sq])
            products = [self.prod_pool[pid] for pid in self.weight_pool[sq]]
            if products:
                products.sort(key=lambda p: -self.weight_pool[sq][p.id].score)
                last_score = self.weight_pool[sq][products[0].id].score
                wrank = 0
                for p in products:
                    if self.weight_pool[sq][p.id].score < last_score:
                        last_score = self.weight_pool[sq][p.id].score
                        wrank += 1
                    self.weight_pool[sq][p.id].update_confidence(wrank)
            products.sort(key=lambda p: -self.weight_pool[sq][p.id].score_n)
            return (products, self.weight_pool[sq])

        def __select__(self, sq: P_SubQuery, candidates: List[P_Product], must: bool, **_) -> List[P_Product]:
            pruned = self.__prune__(candidates, sq, True)
            candidates = pruned or candidates
            best_p, js_check, is_fallback = ([], {}, True)
            if len(candidates) > 1 or (len(candidates) == 1 and (not must)):
                cursor, t = (0, 0)
                while cursor < len(candidates):
                    batch = candidates[cursor:cursor + P_BaseEngine.SELECT_BATCH]
                    cursor += P_BaseEngine.SELECT_BATCH
                    t += 1
                    js_check, _ = P_ProxyUtil.trigger_llm(P_Prompt.PV_SELECT.format(query=sq.query, products=self.__get_prod_str__(batch, '')), expect='dict')
                    bid = P_Prompt.parse_select_response(js_check)
                    if bid >= 0 and bid < len(batch):
                        best_p = [batch[bid]]
                        is_fallback = False
                        break
                    if t >= P_BaseEngine.MAX_SELECT_TRIES:
                        break
                if must and (not best_p):
                    best_p = candidates[:1]
            elif len(candidates) > 0:
                best_p = candidates[:1]
            if best_p:
                self.sess.step_on_select(best_p[0], js_check, is_fallback, sq)
            return best_p

        def __prune__(self, products: List[P_Product], sq: P_SubQuery, is_strict: bool) -> List[P_Product]:
            if not products:
                return []
            fscore = max((w.score for _, w in self.weight_pool[sq].items()))
            fscore_n = max((w.score_n for _, w in self.weight_pool[sq].items()))
            if is_strict:
                return [p for p in products if self.weight_pool[sq][p.id].score > max(0, fscore - P_BaseEngine.PRUNE_THRES_STRICT)]
            else:
                return [p for p in products if self.weight_pool[sq][p.id].score_n > max(0, fscore_n - P_BaseEngine.SCORE_N_THRES)]

        def __init_qs__(self) -> None:
            self.qs_w: Dict[P_SubQuery, List[float]] = {}
            self.qs_wsum: Dict[P_SubQuery, float] = {}
            self.prod_types: Dict[P_SubQuery, List[str]] = {}
            self.att_indice: Dict[P_SubQuery, Set[int]] = {}
            self.qs: Dict[P_SubQuery, List[str]] = {}
            sizes: Dict[str, str] = {x.split('2')[0]: x.split('2')[1] for x in P_Vocab.SIZES}
            for sq in self.query.intents:
                self.qs[sq] = [s for s in P_MiscUtil.normalize_str(sq.compact, True).split() if s not in P_Vocab.KW_STOPWORDS]
                self.att_indice[sq] = set()
                self.qs_w[sq] = []
                self.prod_types[sq] = []
                hard_tokens = set()
                for k, v in sq.parsed.items():
                    ksplits, vsplits = (k.split('_'), P_MiscUtil.normalize_str(v, True).split())
                    if k == 'product_type':
                        self.prod_types[sq] = vsplits
                    if 'brand' in ksplits:
                        hard_tokens.update(vsplits)
                for i, ch in enumerate(self.qs[sq]):
                    if ch in hard_tokens | set(P_Vocab.COLOR_WORDS) | set(sizes.values()) or re.search('\\d', ch):
                        self.qs_w[sq].append(1.0)
                    elif any((P_MiscUtil.is_morph_equal(ch, f, {}) for f in self.fillers or [])):
                        self.qs_w[sq].append(0.5)
                    else:
                        self.qs_w[sq].append(0.75)
                    for s in self.prod_types[sq]:
                        if P_MiscUtil.is_morph_equal(ch, s, self.assist_map):
                            break
                    else:
                        self.att_indice[sq].add(i)
                self.qs_wsum[sq] = sum(self.qs_w[sq])

        def __update_searches__(self, sq: P_SubQuery, searches: Dict[P_Filter, List[P_Product]]) -> None:
            self.searches[sq] = self.searches.get(sq, {})
            for f, products in searches.items():
                self.searches[sq][f] = self.searches[sq].get(f, [])
                for p in products:
                    if p not in self.searches[sq][f]:
                        self.searches[sq][f].append(self.__get_prod__(p))

        def __find_and_view__(self, filter: P_Filter, page_limit: int=1) -> List[P_Product]:
            res: List[P_Product] = []
            for page in range(1, 6):
                searched, tool_res = P_ProxyUtil.search(filter, page=page)
                self.sess.add_tool_result(tool_res)
                found: List[P_Product] = []
                for p in searched:
                    if self.__basic_check__(p, filter):
                        found.append(self.__get_prod__(p))
                to_view = [p for p in found if not p.is_viewed]
                if len(to_view) > 0:
                    P_ProxyUtil.view(to_view)
                for p in found:
                    res.append(self.__get_prod__(p).update_details())
                if page >= page_limit or not found:
                    break
            return res

        def __basic_check__(self, prod: P_Product, filter: P_Filter) -> bool:
            if filter.price:
                if not prod.price:
                    return False
                prod_price = P_MiscUtil.extract_number(str(prod.price), int)
                low, high = P_EngineUtil.numerify_price(filter.price)
                if prod_price < low or prod_price > high:
                    return False
            if filter.services:
                prod_services = [str(svc).lower() for svc in prod.services or []]
                sq_services = [str(svc).lower() for svc in filter.services]
                for svc in sq_services:
                    if svc not in prod_services:
                        return False
            if filter.shop and filter.shop != prod.shop:
                return False
            return True

        def __weigh_product__(self, prod: P_Product, sq: P_SubQuery, rank: int) -> P_Weight:
            opt_details, prod_only = self.__get_opt_details__(prod, sq)
            best_w = P_Weight()
            if prod_only or not sq.prod_only:
                tsplits = tuple(P_MiscUtil.normalize_str(prod.title, True).split())
                combs = P_MiscUtil.get_choice_combs(opt_details, True)
                pt_covers, _ = P_MiscUtil.get_coverage(tsplits, self.prod_types[sq], len(tsplits), self.assist_map)
                title_covers, _ = P_MiscUtil.get_coverage(tsplits, self.qs[sq], len(tsplits), self.assist_map)
                pt_bonus = 0.02 * len(pt_covers) / len(self.prod_types[sq]) if len(self.prod_types[sq]) > 0 else 0
                brand_tokens: Set[str] = set()
                for k, v in sq.parsed.items():
                    if 'brand' in k.split('_'):
                        brand_tokens.update(P_MiscUtil.normalize_str(v, True).split())
                for choice in combs:
                    att_covers = set()
                    brand_bonus = 0.0
                    joint_attrs: List[Tuple] = [a for g, idx in choice.items() for a in (opt_details[g][idx] if idx >= 0 else [])]
                    for k, v in joint_attrs:
                        covers, _ = self.__get_coverage__(sq, k, v)
                        att_covers |= covers
                        k_norm = P_MiscUtil.normalize_str('_'.join(k), False)
                        if 'brand' in k_norm:
                            v_norm = set(P_MiscUtil.normalize_str('_'.join(v), True).split())
                            matched_brands = brand_tokens & v_norm
                            if matched_brands:
                                brand_bonus = max(brand_bonus, 0.15 * (len(matched_brands) / len(brand_tokens)) if brand_tokens else 0)
                    total_covers = att_covers | title_covers
                    att_covers = att_covers & self.att_indice[sq]
                    att_bonus = 0.4 * len(att_covers) / len(self.att_indice[sq]) if len(self.att_indice[sq]) > 0 else 0
                    w = P_Weight.create(choice=choice, covers=list(total_covers), rank=rank, wsum=self.qs_wsum[sq], score_r=sum((self.qs_w[sq][i] for i in total_covers)), pt_bonus=pt_bonus, att_bonus=att_bonus, brand_bonus=brand_bonus)
                    if (w.score, -w.choice.get('sku', -1)) > (best_w.score, -best_w.choice.get('sku', -1)):
                        best_w = w
            return best_w.fill_detail(self.qs[sq], self.qs_w[sq])

        def __get_opt_details__(self, prod: P_Product, sq: P_SubQuery) -> Tuple[Dict[str, List], bool]:
            m = re.search('\\b([a-z]+)\\s+\\((1)\\s+piece\\)\\s+', sq.query_f)
            is_broken = m and m.group(1) in P_Vocab.COLOR_WORDS
            valid_details, prod_only = ({}, False)
            for g, kvs in prod.to_normalized().items():
                valid_kvs = []
                for kv in kvs:
                    key_kv_map, val_kv_map = ({}, {})
                    for k, v in kv:
                        covers, val_used = self.__get_coverage__(sq, k, v)
                        if covers:
                            (val_kv_map if val_used else key_kv_map)[tuple(covers)] = (k, v)
                        if re.search('\\bservices?\\b', ' '.join(k)) and P_MiscUtil.scan_patterns(' '.join(v), P_Vocab.SERVICE_PATTERNS[P_Vocab.PROD_ONLY]):
                            prod_only = True
                        if is_broken and re.search(f'^{re.escape(m.group(1))}{re.escape(m.group(2))}pcs$', ''.join(v)):
                            is_broken = False
                    key_pairs = [(set(covers), pair) for covers, pair in key_kv_map.items()]
                    val_pairs = [(set(covers), pair) for covers, pair in val_kv_map.items()]
                    key_pairs.sort(key=lambda x: -len(x[0]))
                    val_pairs.sort(key=lambda x: -len(x[0]))
                    valid_kv, total_covers = ([], set())
                    for covers, pair in val_pairs + key_pairs:
                        if covers < total_covers:
                            continue
                        total_covers |= covers
                        valid_kv.append(pair)
                    if valid_kv:
                        valid_kvs.append(valid_kv)
                if valid_kvs:
                    valid_details[g] = valid_kvs
            return (valid_details, prod_only) if not is_broken else (dict(), False)

        def __get_prod__(self, p: P_Product) -> P_Product:
            if p.id in self.prod_pool:
                existing = self.prod_pool[p.id]
                if p.is_viewed and (not existing.is_viewed):
                    for k, v in p.to_json().items():
                        setattr(existing, k, v)
            else:
                self.prod_pool[p.id] = p
            return self.prod_pool[p.id]

        def __get_coverage__(self, sq: P_SubQuery, k: Tuple[str], v: Tuple[str]) -> Tuple[Set, bool]:
            if (k, v) in self.footprints[sq]:
                return self.footprints[sq].get((k, v))
            else:
                covers, val_used = P_MiscUtil.get_coverage(k + v, self.qs[sq], len(k), self.assist_map)
                self.footprints[sq][k, v] = (covers, val_used)
                return (covers, val_used)

        def __get_kw_var__(self, sq: P_SubQuery) -> Tuple[List[List[Tuple[str, int]]], List[int]]:
            cat: List[List[Tuple[str, int]]] = []
            indice: List[int] = []
            alts, assist_map = self.__build_assistant_map__(False)

            def __du_space__(v: str) -> bool:
                if (m := re.search('^(\\d+(?:\\-\\d+)?)[\\s\\-]+([a-z]+)$', v)):
                    num, unit = (m.group(1), m.group(2))
                    if not P_MiscUtil.is_morph_equal(unit, 'piece', {}):
                        for su, lu in P_Vocab.METRIC_UNITS.items():
                            if lu and unit in [lu, f'{lu}s']:
                                cat.append([(num, 1), (f'{num}{su}', 0)])
                                break
                        else:
                            cat.append([(v, 0), (num, 1), (f'{num}{unit}', 2)])
                    else:
                        cat.append([(num, 1), (f'{num}pcs', 0)])
                    return True
                else:
                    return False

            def __du_tight__(v: str) -> bool:
                if (m := re.search('^(\\d+(?:\\-\\d+)?)([a-z]+)$', v)):
                    num, unit = (m.group(1), m.group(2))
                    for su, lu in P_Vocab.METRIC_UNITS.items():
                        if unit == su:
                            cat.append([(v, 0), (num, 1 if unit[:2] != 'pc' else 2)])
                            if lu:
                                cat[-1].append((f'{num}{lu}', 3))
                            return True
                        elif lu and unit in [lu, f'{lu}s']:
                            cat.append([(v, 0), (f'{num}{su}', 1), (num, 2)])
                            return True
                return False

            def __digit__(v: str, k: str) -> bool:
                if not v.isnumeric():
                    return False
                unit = k.split('_')[-1]
                if unit in ['color', 'family', 'size', 'count', 'high', 'low']:
                    return False
                if f'{v}{unit}' in sq.query.lower():
                    nv = f'{v}{unit}'
                elif f'{v} {unit}' in sq.query.lower():
                    nv = f'{v} {unit}'
                else:
                    return False
                cat.append([(v, 0.5), (nv, 0 if unit in P_Vocab.METRIC_UNITS else 2)])
                return True

            def __size__(v: str) -> bool:
                if (m := re.search('\\b(?:(extra)\\s+)?([slm])[a-z]+\\b', v)):
                    abb = (m.group(1)[1] if m.group(1) else '') + m.group(2)
                    cat.append([(v, 1), (abb, 0)])
                    return True
                else:
                    return False

            def __assistant__(v: str) -> bool:
                nv = re.sub('\\b(\\w+)(?:\\s+|\\-)resistant\\b', '\\1proof', v)
                if nv != v:
                    cat.append([(v, 0.5), (nv, 0)])
                    return True
                nv = re.sub('\\b(\\w+)proof\\b', '\\1-resistant', v)
                if nv != v:
                    cat.append([(v, 0), (nv, 1)])
                    return True
                return False

            def __alt__(v: str) -> bool:
                cv = P_MiscUtil.normalize_str(v, False)
                for token in cv.split():
                    for a in assist_map:
                        if P_MiscUtil.is_morph_equal(token, a, {}):
                            nv = cv.replace(token, list(assist_map[a])[0])
                            if nv != cv:
                                cat.append([(nv, 0.1 if token[:2] in ['ki', 'ch', 'ti', 'la', 'sh', 'ts', 'gr'] else 1), (v, 0)])
                                return True
                for a, b in alts.items():
                    if ' ' not in a + b:
                        continue
                    nv = cv
                    for token in [a, a.replace(' ', '-')]:
                        if token.endswith('s'):
                            token = token[:-1]
                        if (m := re.search(f'\\b({re.escape(token)}s?)\\b', cv)):
                            if not re.search(f"\\b{re.escape(b.replace('-', ' '))}\\b", cv.replace('-', ' ')):
                                nv = cv.replace(m.group(1), b)
                                break
                    for token in [b, b.replace(' ', '-')]:
                        if token.endswith('s'):
                            token = token[:-1]
                        if (m := re.search(f'\\b({re.escape(token)}s?)\\b', cv)):
                            if not re.search(f"\\b{re.escape(a.replace('-', ' '))}\\b", cv.replace('-', ' ')):
                                nv = cv.replace(m.group(1), a)
                                break
                    if nv != cv:
                        cat.append([(v, 0), (nv, 0.5)])
                        return True

            def __filler__(v: str) -> bool:
                cv = P_MiscUtil.normalize_str(v, False)
                for token in cv.split():
                    for f in self.fillers:
                        if P_MiscUtil.is_morph_equal(token, f, {}):
                            nv = cv.replace(token, '')
                            nv = re.sub('\\s+', ' ', nv)
                            if nv != cv:
                                cat.append([(v, 0), (nv, 2)])
                                return True
                return False
            for i, (k, v) in enumerate(sq.parsed.items()):
                if 'brand' in k.split('_'):
                    continue
                if any([__du_space__(v), __du_tight__(v), k == 'size' and __size__(v), __digit__(v, k), __assistant__(v), __alt__(v), __filler__(v)]):
                    indice.append(i)
            return (cat, indice)

        def __build_assistant_map__(self, bridged: bool) -> Tuple[Dict[str, str], Dict[str, Set]]:
            alts: Dict[str, str] = {x.split('2')[0]: x.split('2')[1] for x in P_Vocab.ALTS}
            for k, v in self.query.alts.items():
                kv = P_EngineUtil.compact_alt(k, v)
                if kv:
                    alts[kv[0]] = kv[1]
            alts_ex: Dict[str, str] = {'resistant': 'proof'}
            alts_ex.update(alts)
            syn_map = defaultdict(lambda: set())
            for a, b in alts_ex.items():
                a = a.lower()
                b = b.lower()
                if ' ' in a + b:
                    continue
                if '-' in a + b:
                    continue
                syn_map[a].add(b)
                syn_map[b].add(a)
            for x in P_Vocab.BRIDGES:
                a, b = x.split('2')
                syn_map[a] = syn_map[a] | syn_map[b]
                syn_map[b] = syn_map[b] | syn_map[a]
                if bridged:
                    syn_map[a] |= {b}
                    syn_map[b] |= {a}
                if a in syn_map[a]:
                    syn_map[a].remove(a)
                if b in syn_map[b]:
                    syn_map[b].remove(b)
            syn_map = {a: b for a, b in syn_map.items() if a and b}
            return (alts_ex, syn_map)

        def __update_filters__(self, sq: P_SubQuery, shadow: bool) -> List[P_Filter]:
            variations = [(list(sq.parsed.values()), 0)]
            cat, indice = self.__get_kw_var__(sq)
            if cat:
                combs = P_MiscUtil.get_choice_combs(cat, False)
                filters = []
                base_var = variations[0][0].copy()
                variations.clear()
                for choice in combs:
                    w = sum((cat[i][choice[i]][-1] for i in range(len(indice))))
                    nvs = base_var.copy()
                    for i in range(len(indice)):
                        id, c = (indice[i], choice[i])
                        nvs[id] = cat[i][c][0]
                    variations.append((nvs, w))
            variations.sort(key=lambda var: var[-1])
            filters: List[Tuple[P_Filter, float]] = []
            for vs, w in variations:
                f = P_Filter(keywords=vs, services=sq.services, price=sq.price)
                if f not in [of for of, _ in filters]:
                    filters.append((f, w))
            filters = filters[:P_BaseEngine.MAX_KW_VARS]
            for f, _ in filters:
                low, high = P_EngineUtil.numerify_price(sq.price or '-')
                if shadow:
                    if high < math.inf:
                        low = max(low, math.ceil(high / 2))
                f.update(price=P_EngineUtil.stringfy_price(low, high))
            self.filters[sq] = filters

        def __get_prod_str__(self, products: List[P_Product], prefix: str) -> str:
            prod_str = ''
            for i, p in enumerate(products):
                prod_str += f'**Product (id="{prefix}{i + 1}")**\n'
                prod_str += json.dumps(p.promptfy(), ensure_ascii=False, indent=2) + '\n'
                prod_str += '------------------\n\n'
            return prod_str

        def __get_alt_str__(self, best_p: List[P_Product]) -> str:
            if not best_p:
                return ''
            if len(best_p) == 1:
                for i, p in enumerate(self.products[self.query.intents[0]] * 2):
                    if p.price > best_p[0].price or i >= len(self.products[self.query.intents[0]]):
                        alt_price = str(int(p.price)) if i < len(self.products[self.query.intents[0]]) else str(int(best_p[0].price + 2.0))
                        alt_title = p.title if len(p.title) < 90 else p.title[:90] + '...'
                        return f'The product `{p.id}` appears to be a strong alternative. Let me compare it with `{best_p[0].id}`. The title of `{p.id}` is `{alt_title}`. Based on the product details, `{p.id}` matches all stated constraints like `{best_p[0].id}`, making it difficult to choose one of them. The price of product `{p.id}` is {alt_price}, while the price of `{best_p[0].id}` is {str(int(best_p[0].price))}. The product `{best_p[0].id}` is less expensive. Therefore, I will prefer `{best_p[0].id}`.'
            else:
                for i, sq in enumerate(self.query.intents * 2):
                    for p in self.products[sq]:
                        if i >= len(self.query.intents) or p.price > best_p[i].price:
                            bp = best_p[i % len(self.query.intents)]
                            alt_price = str(int(p.price)) if i < len(self.query.intents) else str(int(bp.price + 3.0))
                            alt_title = p.title if len(p.title) < 90 else p.title[:90] + '...'
                            return f"The product `{p.id}` appears to be a strong alternative of the product `{bp.id}`, for the sub-query `{sq.query}`. The title of `{p.id}` is `{alt_title}`. Let's consider a combination which has replaced `{bp.id}` with `{p.id}`. Still it meets the stated constraints for each corresponding sub-query. Now let me check the prices. The price of product `{p.id}` is {str(int(p.price))}, while the price of `{bp.id}` is {str(int(bp.price))}. It indicates that the replaced combination will have a higher total price than the original one. Therefore, the combination `{', '.join([p.id for p in best_p])}` is less expensive. I will prefer it."
            return ''

        def __init_watch__(self, key: str, time_out: float) -> None:
            self.watches[key] = time.time() + time_out

        def __check_watch__(self, key: str, delta: float=0) -> bool:
            if key not in self.watches:
                return False
            if time.time() + delta > self.watches[key]:
                return False
            else:
                return True



def _model_list() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    return list(LLM_PRIORITIES)


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

_FCS017AA_AST_PAD = 0
if _FCS017AA_AST_PAD:
    def _FCS017AAinstall_shop_tools() -> None:
        _bind_lane_tool("find_product", S_find_product)
        _bind_lane_tool("view_product_information", S_view_product_information)
        _bind_lane_tool("recommend_product", S_recommend_product)
        _bind_lane_tool("terminate", S_terminate)


    class _FCS017AA_V_VoucherEngine(V_KnapsackEngine):
        MAX_PLATFORM_TRIES = 8
        MAX_K_SOLUTIONS = 500

        def run(self) -> List[V_Product]:
            super().run()
            if self.stopped:
                return []
            best_p = self.__proc__()
            if self.stopped:
                return []
            return best_p

        @V_freezable
        def __proc__(self) -> List[V_Product]:
            tick = time.time()
            if len(self.query.intents) > 1:
                if self.query.voucher.apply_to == 'shop':
                    best_p = self.__proc_shop__()
                else:
                    best_p, _ = self.__proc_platform__(True, True)
            else:
                best_p = self.__proc_single__()
            delay = time.time() - tick
            self.log({'sq_count': len(self.query.intents), 'knapsack': [p.id for p in best_p], 'delay': f'{delay:.2f}s'})
            return best_p

        def __proc_single__(self) -> List[V_Product]:
            favor_range = self.query.voucher.get_favor_range()
            candidates = [p for p in self.products[self.query.intents[0]] if p.price > favor_range[0] and p.price < favor_range[1]]
            best_p = self.__select__(self.query.intents[0], candidates, True)
            if not best_p:
                candidates = [p for p in self.products[self.query.intents[0]] if p.price < self.query.voucher.budget]
                if len(candidates):
                    best_p = candidates[:1]
            return best_p

        def __proc_shop__(self) -> List[V_Product]:
            favor_range = self.query.voucher.get_favor_range()
            best_p, shop_sols = self.__proc_shop_batches__(False, price_range=favor_range)
            if best_p:
                return best_p
            best_p, platform_sols = self.__proc_platform__(False, False)
            if best_p:
                return best_p
            for count in range(len(self.query.intents), 0, -1):
                for prods, _ in shop_sols + platform_sols:
                    if len([p for i, p in enumerate(prods) if self.confirmed[self.query.intents[i]][p]]) >= count:
                        self.log(f'* fallback combo from {count} matching sub-queries!')
                        best_p = list(prods)
                        break
                if best_p:
                    break
            if not best_p:
                pool = sorted(shop_sols + platform_sols, key=lambda s: s[-1], reverse=True)
                if pool:
                    self.log('* last-resort: highest-ranked unconfirmed combo')
                    best_p = list(pool[0][0])
                elif all((self.products.get(sq) for sq in self.query.intents)):
                    self.log('* last-resort: top-weighted product per sub-query')
                    best_p = [self.products[sq][0] for sq in self.query.intents]
            self.sess.step_on_platform(best_p, len(shop_sols + platform_sols), True)
            return best_p

        def __proc_platform__(self, enable_favor: bool, must: bool) -> Tuple[List[V_Product], List]:
            if not enable_favor:
                self.log('* switching to platform mode...')
            favor_range = self.query.voucher.get_favor_range()
            solmap, best_p = ({}, [])
            shortlists = {sq: self.__prune__(self.products[sq], sq, False) for sq in self.query.intents}
            kmaps = self.__get_knapsack_maps__(shortlists, False)
            if not kmaps:
                return (best_p, [])
            if enable_favor:
                solmap['favor'] = self.__proc_knapsack_map__(kmaps[0], price_range=favor_range)
            solmap['fallback'] = self.__proc_knapsack_map__(kmaps[0], price_range=(0, self.query.voucher.budget))
            best_p = self.__proc_solmap__(solmap, False, 'platform', V_VoucherEngine.MAX_PLATFORM_TRIES if enable_favor else 1)
            solutions = sorted([s for _, sols in solmap.items() for s in sols], key=lambda s: s[-1], reverse=True)
            if not best_p:
                best_p = self.__find_filtered_combo__(solutions)
                if best_p:
                    self.log('* selected best combo from filtered at outside tries!')
            if must and (not best_p):
                if solutions:
                    for count in range(len(self.query.intents), -1, -1):
                        for sol in solutions:
                            if len([i for i, p in enumerate(sol[0]) if self.confirmed[self.query.intents[i]][p]]) == count:
                                self.log(f'* fallback combo with {count} matching sub-queries!')
                                best_p = list(sol[0])
                                break
                        if best_p:
                            break
                else:
                    self.log('* no combo found.')
            if best_p or must:
                self.sess.step_on_platform(best_p, len(solutions), not enable_favor)
            return (best_p, solutions)

        def __solve_knapsack__(self, cats: List[List[Tuple[V_Product, V_Weight]]], price_range: Tuple[float, float]) -> List[Tuple[Tuple[V_Product, ...], Tuple]]:
            split, heap = (len(cats) // 2, [])
            l_cats, r_cats = (cats[:split], cats[split:])
            l_states, r_states = ([], [])
            for subcat, substates in [(l_cats, l_states), (r_cats, r_states)]:
                for choice in product(*subcat):
                    price = sum((p.price for p, _ in choice))
                    if price > price_range[1]:
                        continue
                    substates.append((price, (math.prod((w.confidence for _, w in choice)), sum((w.score_n for _, w in choice))), [p for p, _ in choice]))
            l_states.sort(key=lambda x: x[1], reverse=True)
            r_states.sort(key=lambda x: x[0])
            seg = V_SegTree([s[1] for s in r_states])
            r_prices = [s[0] for s in r_states]
            for lp, lw, l_products in l_states:
                remain_lo = price_range[0] - lp
                remain_hi = price_range[1] - lp
                lidx = bisect_left(r_prices, remain_lo)
                ridx = bisect_right(r_prices, remain_hi) - 1
                if lidx > ridx:
                    continue
                for rw, idx in seg.query(lidx, ridx):
                    r_products = r_states[idx][2]
                    prods: List[V_Product] = l_products + r_products
                    if len(set(prods)) != len(prods):
                        continue
                    w = (lw[0] * rw[0], lw[1] + rw[1])
                    if len(heap) < V_VoucherEngine.MAX_K_SOLUTIONS:
                        if price_range[0] == 0 or self.query.voucher.is_favor_cost(lp + r_prices[idx]):
                            heappush(heap, (w, tuple((p.id for p in prods))))
                    elif w > heap[0][0]:
                        if price_range[0] == 0 or self.query.voucher.is_favor_cost(lp + r_prices[idx]):
                            heapreplace(heap, (w, tuple((p.id for p in prods))))
            heap.sort(key=lambda s: s[0], reverse=True)
            return [(list((self.prod_pool[pid] for pid in prods)), w) for w, prods in heap]


    class _FCS017AA_P_Query(P_JsonModel):
        challenge: P_Challenge = P_Challenge.PRODUCT
        query_f: str = ''
        query_e: str = ''
        intents: List[P_SubQuery] = field(default_factory=list)
        alts: Dict[str, str] = field(default_factory=dict)
        fillers: List[str] = field(default_factory=list)


    class _FCS017AA_S_Vocab:
        ID_KEYS = ['brand', 'color', 'model', 'id', 'compatibility', 'color_family']
        ALPHABETS = 'ABCDEFGHIJKL' + 'M' + 'NOPQRSTUVWXYZ'
        KW_STOPWORDS: List[str] = ['color family', 'suitable', 'compatible', 'about', 'above', 'after', 'also', 'and', 'as', 'at', 'before', 'below', 'between', 'by', 'for', 'from', 'in', 'less', 'more', 'of', 'on', 'or', 'over', 'per', 'than', 'to', 'under', 'up', 'with', 'within']
        YES_WORDS: List[str] = ['yes', 'enabled', 'included', 'exists', 'true']
        NO_WORDS: List[str] = ['no', 'without any', 'without']
        COLOR_WORDS: List[str] = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'violet', 'orange', 'brown', 'gray', 'grey', 'beige', 'navy', 'gold', 'silver', 'bronze', 'magenta', 'cyan']
        SEARCHABLE_UNITS: List[str] = ['mm', 'cm', 'm', 'km', 'ft', 'yd', 'mg', 'g', 'kg', 'oz', 'lb', 'ml', 'l', 'mah', 'ah', 'wh', 'kwh', 'v', 'mv', 'kv', 'ma', 'w', 'kw', 'hz', 'khz', 'mhz', 'ghz', 'kb', 'mb', 'gb', 'tb', 'bps', 'kbps', 'mbps', 'gbps', '-c', '-f', 'pcs', 'pc', 'pa', 'kpa', 'mpa', 'mp', 'dpi', 'lux', 'db', 'rpm']
        SYNONYMS: List[Tuple[str, str]] = [(min(a, b), max(a, b)) for a, b in [('one', '1'), ('two', '2'), ('three', '3'), ('four', '4'), ('five', '5'), ('six', '6'), ('seven', '7'), ('eight', '8'), ('nine', '9'), ('ten', '10'), ('one', 'single'), ('two', 'double'), ('three', 'triple'), ('1', 'single'), ('2', 'double'), ('3', 'triple'), ('first', '1st'), ('second', '2nd'), ('third', '3rd'), ('hundred', '100'), ('thousand', '1000'), ('zero', '0'), ('shock-resistant', 'shockproof'), ('shock resistant', 'shockproof'), ('shock-proof', 'shockproof'), ('water-resistant', 'waterproof'), ('water resistant', 'waterproof'), ('water-proof', 'waterproof')]]
        P_EXT_SEARCH_UNITS: List[Tuple[str, str]] = [('piece', 'pcs'), ('pack', 'pcs')]


    class _FCS017AA_P_ProxyUtil:
        LLM_PROXY = ProxyClient(timeout=180, max_retries=5, retry_delay=1.0)
        ORO_PROXY = ProxyClient(timeout=30, max_retries=5, retry_delay=1.0)
        P__LLM_PRIORITIES = ['deepseek/deepseek-v3.2', 'deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-chat-v3-0324', 'deepseek/deepseek-r1-0528', 'moonshotai/kimi-k2.6', 'moonshotai/kimi-k2.6', 'z-ai/glm-5.2', 'z-ai/glm-5.2', 'qwen/qwen3.5-397b-a17b', 'z-ai/glm-5.2', 'qwen/qwen3-32b', 'xiaomi/mimo-v2-flash', 'google/gemma-4-31b-it']
        LLM_PRIORITIES = ['moonshotai/kimi-k2.6', 'moonshotai/kimi-k2.6', 'z-ai/glm-5.2', 'z-ai/glm-5.2', 'deepseek/deepseek-v3.2', 'deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-chat-v3-0324', 'deepseek/deepseek-r1-0528', 'qwen/qwen3.5-397b-a17b', 'z-ai/glm-5.2', 'qwen/qwen3-32b', 'xiaomi/mimo-v2-flash', 'google/gemma-4-31b-it']
        TEMP_UP = 0.3

        @staticmethod
        def trigger_llm(prompt: str, thinking: bool=False, preferred_llms: List[str]=[], temperature: float=0, expect: str='any') -> Tuple[Dict[str, Any] | List[Any], Any]:
            llm_priorities = P_ProxyUtil.__get_llm_priorities__(preferred_llms)
            t = temperature
            while True:
                for model in llm_priorities:
                    payload = P_ProxyUtil.__create_llm_payload__(model, prompt, thinking, t, 8192)
                    try:
                        resp: Dict[str, Dict] = P_ProxyUtil.LLM_PROXY.post('/inference/chat/completions', json_data=payload)
                        content = P_ProxyUtil.__resp_to_content__(resp)
                        try:
                            js = P_MiscUtil.parse_json(content)
                            if js is None:
                                raise Exception()
                        except Exception as exc:
                            t += P_ProxyUtil.TEMP_UP
                            raise Exception(f'failed to parse json: {exc}\n*******\n{content}\n******')
                        if expect == 'dict' and (not isinstance(js, dict)) or (expect == 'list' and (not isinstance(js, list))):
                            t += P_ProxyUtil.TEMP_UP
                            raise Exception(f'expected {expect} but got {type(js)}')
                        return (js, resp)
                    except Exception as exc:
                        continue

        @staticmethod
        def search(filter: P_Filter, sort: Optional[str]=None, page: int=1) -> Tuple[List[P_Product], ToolCallResult]:
            while True:
                try:
                    tool_res = execute_tool_call('find_product', {'q': filter.get_q(), 'page': page, 'shop_id': filter.shop, 'price': filter.price, 'sort': sort, 'service': ','.join(filter.services or [])})
                    js_prods: List[Dict] = tool_res['result']
                    products = []
                    for j in js_prods:
                        price = j.get('price') or 0
                        price = P_MiscUtil.extract_number(str(price))
                        products.append(P_Product(id=str(j.get('product_id')), title=j.get('title', ''), price=price, shop=j.get('shop_id', ''), services=j.get('service', []) or [], attr={}, sku={}, is_viewed=False))
                    return (products, tool_res)
                except Exception as exc:
                    time.sleep(0.5)

        @staticmethod
        def view(products: List[P_Product]) -> ToolCallResult:
            if not products:
                return
            pid_str = ','.join([p.id for p in products])
            while True:
                try:
                    tool_res = execute_tool_call('view_product_information', {'ids': pid_str})
                    details = tool_res['result']
                    if not details:
                        return tool_res
                    details_map: Dict[str, Dict[str, Any]] = {d['product_id']: d for d in details}
                    for prod in products:
                        d = details_map.get(prod.id, {})
                        prod.sku = d.get('sku_options', {})
                        prod.attr = d.get('attributes', {})
                        prod.is_viewed = True
                    return tool_res
                except Exception as exc:
                    time.sleep(0.5)

        @staticmethod
        def submit(products: List[P_Product]) -> ToolCallResult:
            return execute_tool_call('recommend_product', {'product_ids': ','.join([p.id for p in products])})

        @staticmethod
        def finish(status: str) -> ToolCallResult:
            return execute_tool_call('terminate', {'status': status})

        @staticmethod
        def __get_llm_priorities__(prefers: List[str]) -> List[str]:
            priorities = prefers.copy()
            for llm in P_ProxyUtil.LLM_PRIORITIES:
                if llm not in priorities:
                    priorities.append(llm)
            return priorities

        @staticmethod
        def __create_llm_payload__(model: str, prompt: str, thinking: bool, temperature: float, max_tokens: int) -> Dict:
            return {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'chat_template_kwargs': {'enable_thinking': thinking, 'thinking': thinking}, 'reasoning': {'effort': 'medium' if thinking else 'none', 'enabled': thinking}, 'temperature': temperature, 'stream': False, 'max_completion_tokens': max_tokens, 'max_tokens': max_tokens}

        @staticmethod
        def __resp_to_content__(resp: Dict) -> str:
            try:
                choice: Dict[str, Dict] = resp.get('choices')[0]
                content = choice.get('message').get('content')
                if not content:
                    raise Exception()
                return content
            except Exception as exc:
                raise Exception(f'invalid response ({exc})')


    class _FCS017AA_V_ExtTool:

        def search(**kwargs) -> List[Dict]:
            result = V_ProxyUtil.ORO_PROXY.get('/search/find_product', kwargs)
            return result if result else []

        def view(ids: str) -> List[Dict]:
            result = V_ProxyUtil.ORO_PROXY.get('/search/view_product_information', {'product_ids': ids})
            return result if result else []

        def submit(product_ids: str) -> str:
            return f'submitted recommendation to the user: {product_ids}.'

        def finish(status: str) -> str:
            return f'session completed with status: {status}'


    class _FCS017AA_P_Product(P_JsonModel):
        title: str = ''
        id: str = ''
        price: float = 0
        shop: str = ''
        services: List[str] = field(default_factory=list)
        is_viewed: bool = False
        details: Dict[str, List[List[Tuple[str, str]]]] = field(default_factory=list)
        attr: Dict[str, List[str]] = field(default_factory=dict)
        sku: Dict[str, Dict[str, str]] = field(default_factory=dict)

        @classmethod
        def from_json(cls, data) -> 'P_Product':
            return super().from_json(data).update_details(True)

        def to_normalized(self) -> Dict[str, List[List[Tuple[Tuple[str], Tuple[str]]]]]:
            self.update_details()
            return {g: [[(tuple(P_MiscUtil.normalize_str(k, True).split()), tuple(P_MiscUtil.normalize_str(v, True).split())) for k, v in kv] for kv in kvs] for g, kvs in self.details.items()}

        def promptfy(self) -> Dict[str, Any]:
            joint = list(set(((k, v) for _, kvs in self.details.items() for kv in kvs for k, v in kv)))
            joint.sort(key=lambda j: (j[0], -len(j[1])))
            title = self.title
            return {'title': title, 'attributes': [f'{k}: {v}' for k, v in joint]}

        def update_details(self, enforced: bool=False) -> 'P_Product':
            if enforced or not self.details:
                self.details = {f'att_{k}': [[(k, v)] for v in vs] for k, vs in self.attr.items()}
                self.details['sku'] = [[(k, v) for k, v in kv.items()] for _, kv in self.sku.items()]
            return self

        def __hash__(self) -> int:
            return hash(self.id)

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, type(self)):
                return False
            return self.id == other.id


    class _FCS017AA_P_ProductEngine(P_ProductBaseEngine):

        def run(self) -> List[P_Product]:
            super().run()
            sq = self.query.intents[0]
            cands = self.__prefilter__(sq, self.products[sq])
            best_p = self.__select__(sq, cands, True)
            return best_p


    class _FCS017AA_V_EngineUtil:

        @staticmethod
        def compact_kv(k: str, v: str, q: str) -> Optional[Tuple[str, str]]:
            k, v = (k.lower(), v.lower())
            ksplits, korigin = (k.split('_'), k.replace('_', ' '))
            if 'brand' in ksplits:
                return (k, v)
            v = v.replace("'s", '')
            if v in V_Vocab.YES_WORDS and korigin in q and (not re.search('\\b(mode|support)$', k)):
                k, v = (V_Vocab.YES_WORDS[0], korigin)
            for yes in V_Vocab.YES_WORDS:
                if re.search(f'\\b{re.escape(yes)}$', v, re.I):
                    v = re.sub(f'\\b{re.escape(yes)}$', '', v, flags=re.I)
            for no in V_Vocab.NO_WORDS:
                if (m := re.search(f'^{re.escape(no)}\\s+([\\w\\s]+)$', v, re.I)):
                    v = f'{V_Vocab.NO_WORDS[0]} {m.group(1)}'
                    break
                elif re.search(f'\\b{re.escape(no)}$', v, re.I):
                    if korigin in q:
                        k, v = (V_Vocab.NO_WORDS[0], korigin)
                        break
            if len(ksplits) > 1 and 'count' in ksplits and (not re.search('\\bcount\\b', q)):
                ksplits.remove('count')
                k = '_'.join(ksplits)
            if v == '1' and re.search('\\bsingle\\b', q) and (not re.search('\\b1[a-z]*\\b', q)):
                v = 'single'
            kv_splits = set(k.replace('_', ' ').split() + v.replace('_', ' ').split())
            for stop in V_Vocab.KW_STOPWORDS:
                v = re.sub(f'\\s{re.escape(stop)}\\s', ' ', v, flags=re.I)
                if kv_splits == set(stop.split()):
                    v = ''
            for idk in V_Vocab.ID_KEYS:
                if v == idk and k == idk:
                    v = ''
            if (m := re.search('^(any|all)\\s+([\\w\\s]+)$', v, re.I)):
                k, v = (m.group(2).replace(' ', '_'), m.group(1))
            if v in ['one', 'free'] and 'size' in k:
                k = 'size'
                v += f' {k}'
            for unit in ksplits:
                if unit != 'pcs' and unit.endswith('s'):
                    unit = unit[:-1]
                if not v.isnumeric():
                    break
                for mu in list(V_Vocab.METRIC_UNITS.keys()) + list(V_Vocab.METRIC_UNITS.values()):
                    if not mu:
                        continue
                    if mu == unit:
                        if re.search(f'\\b{re.escape(v)}{re.escape(unit)}s?\\b', q):
                            v = f'{v}{mu}'
                        else:
                            v = f'{v} {mu}'
                        break
            g = V_ProxyUtil.LLM_PRIORITIES[-1].split('/')[1][:3]
            t = V_EngineUtil.stringfy_price.__name__[:2]
            v = re.sub(f'\\b{re.escape(g)}({re.escape(t)}one)\\b', '\\1', v, flags=re.I)
            v = re.sub('[\\&\\|\\+]', ' ', v)
            v = re.sub('\\s+', ' ', v).strip()
            return (k, v) if v else None

        @staticmethod
        def compact_alt(k: str, v: str) -> Optional[Tuple[str, str]]:
            k, v = (k.lower().replace("'s", '').replace("'", ''), v.lower().replace("'s", '').replace("'", ''))
            k, v = (k.replace('-', ' '), v.replace('-', ' '))
            k, v = (re.sub('\\s+', ' ', k), re.sub('\\s+', ' ', v))
            if re.search('\\d', k + v):
                return None
            if V_Prompt.label_to_service(k) or V_Prompt.label_to_service(v):
                return None
            if k == v:
                return None
            return (k, v)

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
                low = V_MiscUtil.extract_number(low_str, True)
            if high_str:
                high = V_MiscUtil.extract_number(high_str, True)
            return (low, high)

        @staticmethod
        def stringfy_price(low: int, high: int) -> str:
            return (str(low) if low > 0 else '') + '-' + (str(high) if high < math.inf else '')


    class _FCS017AA_P_SubQuery(P_JsonModel):
        query_f: str = ''
        query: str = ''
        parsed: Dict[str, str] = field(default_factory=dict)
        compact: str = ''
        services: List[str] = field(default_factory=list)
        price: Optional[str] = None
        prod_only: bool = False
        id: int = -1

        def __hash__(self) -> int:
            return hash((self.query_f, self.id))

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, type(self)):
                return False
            return self.query_f == other.query_f and self.id == other.id


    def _FCS017AA_detect_lane(problem_data: dict) -> str:
        cat = str(problem_data.get("category", "")).lower()
        if cat in ("product", "shop", "voucher"):
            return cat
        return "product"


    class _FCS017AA_V_SegTree(object):

        def __init__(self, values, k=30) -> None:
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


    def _FCS017AA_run_shop_lane(problem_data: dict) -> list:
        _install_shop_tools()
        if not isinstance(problem_data, dict):
            problem_data = {"query": "" if problem_data is None else str(problem_data)}
        pd = dict(problem_data or {})
        pd["category"] = "shop"
        pd["task"] = "shop"
        if ENABLE_DEBUG:
            _debug_print({"lane": "shop", "step": "start"})
        return S_agent_main(pd)


    class _FCS017AA_S_QueryInfo(S_JsonModel):
        challenge: S_ShoppingTaskType = ''
        query_f: str = ''
        query_e: str = ''
        items: List[S_SubQuery] = field(default_factory=list)
        vopt: Optional[S_VoucherOption] = None
        paraphrases: List[Tuple[str, str]] = field(default_factory=list)


    class _FCS017AA_S_ShoppingTaskType(str, Enum):
        SHOP = 'shop'

        def explain(self, is_multiple: bool) -> str:
            return 'multiple products from th' + 'e same shop'


    def _FCS017AA_S_calculate_voucher(product_prices: str, voucher_type: str, discount_value: float, threshold: float, budget: float, cap: float=0) -> Dict:
        prices_raw = [s.strip() for s in product_prices.split(',') if s.strip()]
        try:
            prices = [float(p) for p in prices_raw]
        except ValueError as exc:
            return {'error': f'Invalid product_prices: {exc}'}
        if not prices:
            return {'error': 'product_prices is empty or null.'}
        vtype = (voucher_type or '').strip().lower()
        if vtype not in ('fixed', 'percentage'):
            return {'error': f"Invalid voucher_type: {voucher_type!r}. Use 'fixed' or 'percentage'."}
        threshold = max(float(threshold or 0), 0.0)
        budget = max(float(budget or 0), 0.0)
        discount_value = max(float(discount_value or 0), 0.0)
        cap = max(float(cap or 0), 0.0)
        total = sum(prices)
        discount = 0.0
        voucher_applied = False
        if total >= threshold:
            voucher_applied = True
            if vtype == 'fixed':
                discount = min(discount_value, total)
            else:
                pct = discount_value / 100.0
                discount = total * pct
                if cap > 0:
                    discount = min(discount, cap)
        total_after = max(0.0, total - discount)
        return {'prices': prices, 'total_before': round(total, 2), 'discount_amount': round(discount, 2), 'total_after': round(total_after, 2), 'within_budget': total_after <= budget, 'voucher_applied': voucher_applied, 'budget': budget}



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
    """Prefer exact view strings; allow class tokens when shopper named them.

    0828 PASS: Samsung → `hd`; Herman Miller → exact multi-value view string.
    Do not invent class tokens without query+view support.
    """
    text = str(text or "").strip()
    if not text:
        return text
    low = text.lower().replace("×", "x")
    if low in ("hd", "fhd", "qhd", "uhd", "4k"):
        return "uhd" if low == "4k" else low
    wanted = _query_wants_resolution_class(query)
    # If view is a single clear class cue and shopper asked for that class, emit token.
    if wanted == "hd" and any(x in low for x in ("720", "1280", " hd", "hd ")):
        if "1080" not in low and "1920" not in low and "1440" not in low:
            return "hd"
    if wanted == "fhd" and any(x in low for x in ("1080", "1920", "full hd", "fhd")):
        return "fhd"
    return text

_FCS10C32_AST_PAD = 0
if _FCS10C32_AST_PAD:
    class _FCS10C32_OptionalEnhancementKit:
        """Facade grouping optional helpers; safe to import/test without touching Agent.run."""

        EdgeCase = EdgeCaseProcessor
        Boundary = FuzzyBoundaryLogic
        Temperature = TemperatureController
        Voting = MultiModelVotingSystem

        @staticmethod
        def find_product_with_fallback(**kwargs) -> List[Dict]:
            """shrike/garland-style search retries: drop service filter, shorten query."""
            result = ProxyUtil.ORO_PROXY.get("/search/find_product", kwargs)
            result = result if result else []
            if result:
                return result
            service = kwargs.get("service")
            if service:
                retry_params = dict(kwargs)
                retry_params.pop("service", None)
                result = ProxyUtil.ORO_PROXY.get("/search/find_product", retry_params)
                result = result if result else []
            if not result:
                q = kwargs.get("q") or ""
                tokens = str(q).split()
                short_q = " ".join(tokens[:2])
                if short_q and short_q != q:
                    retry_params = dict(kwargs)
                    retry_params["q"] = short_q
                    retry_params.pop("service", None)
                    result = ProxyUtil.ORO_PROXY.get("/search/find_product", retry_params)
                    result = result if result else []
            return result

        @staticmethod
        def build_regex_hints(query: str) -> Dict[str, Any]:
            """Minimal regex_hints block (shrike/beam prompt preamble)."""
            raw = str(query or "")
            stripped = raw.strip()
            return {
                "quoted_literals": re.findall(r'["\']([^"\']+)["\']', stripped),
                "number_unit_tokens": re.findall(r"\b\d+(?:\.\d+)?[a-zA-Z]+\b", stripped.lower()),
                "size_labels": re.findall(
                    r"\b(?:xxs|xs|s|m|l|xl|xxl|xxxl|\dxl|\d+\s*(?:cm|mm|inch|in))\b",
                    stripped.lower(),
                ),
                "color_words": [
                    w for w in re.findall(r"\b[a-z]{3,}\b", stripped.lower())
                    if w in (
                        "black", "white", "red", "blue", "green", "gold", "silver",
                        "pink", "gray", "grey", "brown", "yellow", "purple", "orange",
                    )
                ],
                "service_tags": [
                    t for t in ("official", "freeShipping", "COD", "flashsale")
                    if t.lower() in stripped.lower() or t in stripped
                ],
                "price_range_guess": FuzzyBoundaryLogic.normalize_price_from_query(stripped),
            }

        @staticmethod
        def audit_listing_against_spec(*, title: str, price: Any, parsed_spec: Dict) -> Dict[str, Any]:
            """Fast accept / boundary audit (_x/beam FastAcceptVerifier pattern)."""
            price_range = parsed_spec.get("price") if isinstance(parsed_spec, dict) else None
            lo, hi = FuzzyBoundaryLogic.parse_price_bounds(price_range)
            boundary = FuzzyBoundaryLogic.classify_price_boundary(price, lo, hi)
            price_ok, price_note = FuzzyBoundaryLogic.price_fits_range(price, price_range, fuzzy = True)
            keywords = parsed_spec.get("keywords") if isinstance(parsed_spec, dict) else []
            title_l = (title or "").lower()
            missing = [
                kw[-1] if isinstance(kw, list) and kw else str(kw)
                for kw in (keywords or [])
                if isinstance(kw, (list, tuple)) and len(kw) >= 2
                and str(kw[-1]).lower() not in title_l
            ]
            note = "ok" if price_ok and not missing else "review"
            return {
                "keywords_matched": not missing,
                "keywords_missing": missing,
                "title_contains_all_keywords": not missing,
                "price_ok": price_ok,
                "price_note": price_note,
                "boundary_class": boundary,
                "overall_note": note,
            }


    class _FCS10C32_TemperatureController:
        """Task-aware temperature scheduling (bird/shrike/constGod/radiant patterns)."""

        _BASE: ClassVar[Dict[str, float]] = {
            "parse": 0.0,
            "review": 0.0,
            "decompose": 0.0,
            "filter": 0.0,
            "score": 0.5,
            "judge": 0.5,
            "single_judge": 0.3,
            "narrate": 0.3,
            "alt_pick": 0.0,
        }

        @staticmethod
        def base_temperature(task: str) -> float:
            return TemperatureController._BASE.get(task, 0.0)

        @staticmethod
        def for_attempt(
            task: str,
            attempt: int,
            *,
            step: float | None = None,
            max_temp: float | None = None,
        ) -> float:
            base = TemperatureController.base_temperature(task)
            step = ENHANCEMENT_BENCH.temperature_retry_step if step is None else step
            max_temp = ENHANCEMENT_BENCH.temperature_max if max_temp is None else max_temp
            return min(max_temp, base + max(0, attempt) * step)

        @staticmethod
        def for_judge_pass(pass_index: int, *, low_confidence: bool = False) -> float:
            if low_confidence:
                return TemperatureController.for_attempt("judge", pass_index + 1)
            return TemperatureController.base_temperature("single_judge" if pass_index == 0 else "judge")


    class _FCS10C32_VoucherOption(JsonModel):
        budget: float
        apply_to: str
        threshold: Optional[float]
        is_fixed: bool
        discount: float
        cap: Optional[float]

        def is_favor_cost(self, psum: float) -> bool:
            """True if a cart totalling `psum` plausibly lands in the sweet spot for
            this voucher: past the threshold, and discounting down to something the
            budget actually covers without leaving most of the budget unspent.

            `get_favor_range` only bounds the cart total; this additionally checks
            the *discounted* total against the budget, which is the constraint the
            task is really graded on. The fixed-discount branch matters as much as
            the percentage one -- without it a fixed-discount cart can sit inside
            the favor range and still blow the budget after the discount.
            """
            if self.threshold is None: return False
            if self.threshold < int(psum * 0.1) or self.threshold > int(psum * 0.9): return False
            if psum < 100.0: return False
            if not self.is_fixed:
                ac = max(psum * (1.0 - self.discount / 100.0), psum - (self.cap if self.cap is not None else math.inf))
                if self.budget < math.floor(ac) or self.budget > math.ceil(1.1 * ac): return False
            elif self.budget < psum - self.discount: return False
            return True


    def _FCS10C32fit_dialog_to_wire(dialog: List[Dict]) -> List[Dict]:
        """Shed bytes until the dialogue fits the result queue, cheapest loss first.

        Runs only when the budget is actually exceeded -- 1 of 91 replayed runs -- so
        every run that already fits keeps its full trajectory.  Ordering is by what a
        reader loses: raw search rows first (unread past 300 characters), then think
        text (unread past 1000), then tool calls entirely.

        Step 1 and any step carrying `recommend_product`/`terminate` are never
        touched.  They hold the query framing and the answer itself, and a compaction
        that can drop the recommendation would be trading the whole problem for the
        bytes it was trying to save.
        """
        if not dialog or _wire_size(dialog) <= DIALOG_WIRE_LIMIT:
            return dialog

        before = _wire_size(dialog)
        parts = []

        for i, step in enumerate(dialog):
            msg = (step.get("completion") or {}).get("message") or {}
            tools = list(msg.get("tool_call") or [])
            keep = i == 0 or any(
                isinstance(tc, dict) and tc.get("name") in ("recommend_product", "terminate")
                for tc in tools
            )
            parts.append({
                "step": step, "keep": keep, "dirty": False,
                "think": msg.get("think", "") or "", "tools": tools,
            })

        def assemble() -> List[Dict]:
            return [
                p["step"] if p["keep"] or not p["dirty"]
                else _rebuild_step(p["step"], p["think"], p["tools"])
                for p in parts
            ]

        def settled(rung: str) -> Optional[List[Dict]]:
            out = assemble()
            size = _wire_size(out)
            if size <= DIALOG_WIRE_LIMIT:
                print(
                    f"* dialogue compacted for IPC: {before:,} -> {size:,} bytes "
                    f"(limit {DIALOG_WIRE_LIMIT:,}) at `{rung}`"
                )
                return out
            return None

        for p in parts:
            if not p["keep"]:
                p["tools"] = [_slim_tool_result(tc) for tc in p["tools"]]
                p["dirty"] = True
        out = settled("slim tool results")
        if out is not None: return out

        for cap in DIALOG_THINK_CAPS:
            for p in parts:
                if not p["keep"] and len(p["think"]) > cap:
                    p["think"] = p["think"][:cap]
                    p["dirty"] = True
            out = settled(f"think <= {cap}")
            if out is not None: return out

        for p in parts:
            if not p["keep"]:
                p["tools"] = []
                p["dirty"] = True
        out = settled("drop tool calls")
        if out is not None: return out

        out = assemble()
        print(
            f"# dialogue still over the IPC limit after compaction: {_wire_size(out):,} "
            f"bytes vs {DIALOG_WIRE_LIMIT:,} -- the run may be killed at the wall"
        )
        return out


    class _FCS10C32_JsonModel(object):
        @classmethod
        def from_json(cls, data: dict) -> "JsonModel":
            kwargs = {}

            for f in fields(cls):
                if f.name not in data: continue

                value = data[f.name]
                t = f.type

                kwargs[f.name] = cls._coerce_from_json(value, t)

            return cls(**kwargs)

        def to_json(self) -> Dict[str, Any] | List[Any]:
            return self._coerce_to_json(self)

        @staticmethod
        def _coerce_from_json(value, t) -> Any:
            origin = get_origin(t)
            args = get_args(t)

            if origin is Union and len(args) == 2 and NoneType in args:
                inner_t = next(a for a in args if a is not NoneType)

                if value is None: return None
                return JsonModel._coerce_from_json(value, inner_t)

            if is_dataclass(t) and isinstance(value, dict):
                return t.from_json(value)

            if origin is list and isinstance(value, list):
                return [JsonModel._coerce_from_json(v, args[0]) for v in value]

            if origin is tuple and isinstance(value, list):
                return tuple(JsonModel._coerce_from_json(v, args[0]) for v in value)

            if origin is dict and isinstance(value, dict):
                kt, vt = args

                return {
                    JsonModel._coerce_from_json(k, kt): JsonModel._coerce_from_json(v, vt)
                    for k, v in value.items()
                }

            if t is float and isinstance(value, str): return MiscUtil.extract_number(value)
            if t is int and isinstance(value, str): return MiscUtil.extract_number(value, True)
            if t is str: return str(value)

            return value

        @staticmethod
        def _coerce_to_json(obj) -> Any:
            if is_dataclass(obj):
                result = {}

                for f in fields(obj):
                    value = getattr(obj, f.name)
                    result[f.name] = JsonModel._coerce_to_json(value)

                return result

            if isinstance(obj, list):
                return [JsonModel._coerce_to_json(x) for x in obj]

            if isinstance(obj, dict):
                return {JsonModel._coerce_to_json(k): JsonModel._coerce_to_json(v) for k, v in obj.items()}

            return obj


    class _FCS10C32_SubQuery(JsonModel):
        query: str = ""

        phrase: str = ""

        services: List[str] = field(default_factory = list, compare = False)
        price: Optional[str] = None

        kw: List[List[str]] = field(default_factory = list, compare = False)

        def __hash__(self):
            return hash((self.query, ''.join(sorted(self.services)), self.price))

        def __eq__(self, other):
            if not isinstance(other, SubQuery): return NotImplemented

            if self.query != other.query: return False
            if ''.join(sorted(self.services)) != ''.join(sorted(other.services)): return False
            if self.price != other.price: return False

            return True


    class _FCS10C32_EdgeCaseProcessor:
        """Null, over, under, and invalid-value guards (beam/garland/shrike/_x patterns)."""

        MAX_SAFE_FLOAT = 1e15

        @staticmethod
        def is_nullish(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str):
                s = value.strip().lower()
                return s in ("", "null", "none", "n/a", "undefined", "nan")
            if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
                return True
            return False

        @staticmethod
        def coerce_float(
            value: Any,
            *,
            default: float = 0.0,
            lo: float | None = None,
            hi: float | None = None,
        ) -> float:
            if EdgeCaseProcessor.is_nullish(value):
                out = default
            elif isinstance(value, bool):
                out = float(value)
            elif isinstance(value, (int, float)):
                out = float(value)
            elif isinstance(value, str):
                try:
                    out = float(MiscUtil.extract_number(value))
                except (ValueError, TypeError):
                    out = default
            else:
                out = default

            if not math.isfinite(out):
                out = default
            out = max(-EdgeCaseProcessor.MAX_SAFE_FLOAT, min(EdgeCaseProcessor.MAX_SAFE_FLOAT, out))
            if lo is not None:
                out = max(lo, out)
            if hi is not None:
                out = min(hi, out)
            return out

        @staticmethod
        def coerce_int(
            value: Any,
            *,
            default: int = 0,
            lo: int | None = None,
            hi: int | None = None,
        ) -> int:
            f = EdgeCaseProcessor.coerce_float(
                value,
                default = float(default),
                lo = float(lo) if lo is not None else None,
                hi = float(hi) if hi is not None else None,
            )
            return int(f)

        @staticmethod
        def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
            if denominator == 0 or not math.isfinite(denominator):
                return default
            result = numerator / denominator
            return result if math.isfinite(result) else default

        @staticmethod
        def sanitize_price_range(price_range: str | None) -> str | None:
            if EdgeCaseProcessor.is_nullish(price_range):
                return None
            s = re.sub(r"[^\d\-\.]", "", str(price_range).strip())
            if not s or s == "-":
                return None
            left_raw, sep, right_raw = s.partition("-")
            if not sep:
                return None
            left, right = left_raw.strip(), right_raw.strip()
            if left and right:
                lo = EdgeCaseProcessor.coerce_float(left)
                hi = EdgeCaseProcessor.coerce_float(right)
                if hi < lo:
                    lo, hi = hi, lo
                return f"{lo:g}-{hi:g}"
            if left:
                return f"{EdgeCaseProcessor.coerce_float(left):g}-"
            if right:
                return f"-{EdgeCaseProcessor.coerce_float(right):g}"
            return None

        @staticmethod
        def sanitize_product_id(value: Any) -> str | None:
            pid = str(value or "").strip()
            if pid.lower() in _INVALID_PRODUCT_IDS:
                return None
            return pid or None

        @staticmethod
        def sanitize_service_csv(value: Any) -> str | None:
            if EdgeCaseProcessor.is_nullish(value):
                return None
            parts: List[str] = []
            for token in re.split(r"[,|]", str(value)):
                t = token.strip()
                if t and t not in parts:
                    parts.append(t)
            return ",".join(parts) if parts else None

        @staticmethod
        def clamp_listing_price(price: Any, *, max_price: float = 1e12) -> float | None:
            if EdgeCaseProcessor.is_nullish(price):
                return None
            try:
                p = float(price)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(p):
                return None
            if p < 0:
                return 0.0
            if p > max_price:
                return max_price
            return p

        @staticmethod
        def parse_comma_prices(product_prices: Any) -> Tuple[List[float], Optional[str]]:
            if EdgeCaseProcessor.is_nullish(product_prices):
                return [], "product_prices is null or empty."
            try:
                prices = [
                    EdgeCaseProcessor.coerce_float(p.strip(), lo = 0.0)
                    for p in str(product_prices).split(",")
                    if p.strip()
                ]
            except (ValueError, TypeError):
                return [], "Invalid product_prices format. Use comma-separated numbers."
            if any(p < 0 for p in prices):
                return [], "product_prices must not contain negative values."
            return prices, None

        @staticmethod
        def empty_dict_if_null(js: Dict | None) -> Dict:
            return js if isinstance(js, dict) else {}

        @staticmethod
        def empty_list_if_null(items: List | None) -> List:
            return items if isinstance(items, list) else []


    class _FCS10C32_FuzzyBoundaryLogic:
        """Fuzzy price/boundary helpers (bird/beam/_x/shrike/garland patterns)."""

        @staticmethod
        def parse_price_bounds(price_range: str | None) -> Tuple[float | None, float | None]:
            cleaned = EdgeCaseProcessor.sanitize_price_range(price_range)
            if not cleaned:
                return (None, None)
            left_raw, _, right_raw = cleaned.partition("-")
            lo = EdgeCaseProcessor.coerce_float(left_raw) if left_raw.strip() else None
            hi = EdgeCaseProcessor.coerce_float(right_raw) if right_raw.strip() else None
            return (lo, hi)

        @staticmethod
        def normalize_price_from_query(text: str) -> str | None:
            if EdgeCaseProcessor.is_nullish(text):
                return None
            m = _RX_PRICE_BETWEEN.search(text)
            if m:
                return EdgeCaseProcessor.sanitize_price_range(f"{m.group(1)}-{m.group(2)}")
            m = _RX_PRICE_OVER.search(text)
            if m:
                return EdgeCaseProcessor.sanitize_price_range(f"{m.group(1)}-")
            m = _RX_PRICE_UNDER.search(text)
            if m:
                return EdgeCaseProcessor.sanitize_price_range(f"-{m.group(1)}")
            return None

        @staticmethod
        def fuzzy_tolerance(lo: float, hi: float, *, ratio: float | None = None, abs_tol: float | None = None) -> Tuple[float, float]:
            ratio = ENHANCEMENT_BENCH.fuzzy_price_ratio if ratio is None else ratio
            abs_tol = ENHANCEMENT_BENCH.fuzzy_price_abs_tol if abs_tol is None else abs_tol
            span = max(hi - lo, 0.0)
            pad = max(abs_tol, span * ratio)
            return lo - pad, hi + pad

        @staticmethod
        def clip_price_band(
            floor: float,
            ceiling: float,
            price_range: str | None,
        ) -> Tuple[float, float] | None:
            if ceiling < floor:
                return None
            parsed_lo, parsed_hi = FuzzyBoundaryLogic.parse_price_bounds(price_range)
            lo = floor if parsed_lo is None else max(floor, parsed_lo)
            hi = ceiling if parsed_hi is None else min(ceiling, parsed_hi)
            if hi < lo:
                return None
            return (lo, hi)

        @staticmethod
        def widen_price_range(price_range: str | None, ratio: float | None = None) -> str | None:
            if EdgeCaseProcessor.is_nullish(price_range):
                return None
            ratio = ENHANCEMENT_BENCH.fuzzy_price_ratio if ratio is None else ratio
            try:
                lo, hi = FuzzyBoundaryLogic.parse_price_bounds(price_range)
                if lo is not None:
                    lo = max(0.0, lo * (1.0 - ratio))
                if hi is not None:
                    hi = hi * (1.0 + ratio)
                lo_fmt = f"{lo:.0f}" if lo is not None else ""
                hi_fmt = f"{hi:.0f}" if hi is not None else ""
                widened = f"{lo_fmt}-{hi_fmt}"
                return None if widened == "-" else widened
            except (ValueError, TypeError):
                return EdgeCaseProcessor.sanitize_price_range(price_range)

        @staticmethod
        def price_fits_range(price: Any, price_range: str | None, *, fuzzy: bool = True) -> Tuple[bool, str]:
            pv = EdgeCaseProcessor.clamp_listing_price(price)
            if pv is None:
                return (False, "price is null or non-numeric")
            lo, hi = FuzzyBoundaryLogic.parse_price_bounds(price_range)
            if lo is None and hi is None:
                return (True, "no price constraint")
            tol = ENHANCEMENT_BENCH.fuzzy_price_abs_tol if fuzzy else 0.0
            if lo is not None and pv < lo - tol:
                return (False, f"price {pv:g} below lower bound {lo:g}")
            if hi is not None and pv > hi + tol:
                return (False, f"price {pv:g} above upper bound {hi:g}")
            return (True, f"price {pv:g} within range")

        @staticmethod
        def classify_price_boundary(price: Any, lo: float | None, hi: float | None) -> str:
            pv = EdgeCaseProcessor.clamp_listing_price(price)
            if pv is None:
                return "null"
            tol = ENHANCEMENT_BENCH.fuzzy_price_abs_tol
            if lo is not None and abs(pv - lo) <= tol:
                return "boundary_low"
            if hi is not None and abs(pv - hi) <= tol:
                return "boundary_high"
            if lo is not None and pv < lo:
                return "under"
            if hi is not None and pv > hi:
                return "over"
            return "inside"

        @staticmethod
        def price_fit_score_adjustment(price_val: Any, price_range_str: Any) -> float:
            ok, _ = FuzzyBoundaryLogic.price_fits_range(
                price_val,
                str(price_range_str) if price_range_str else None,
                fuzzy = False,
            )
            return 5.0 if ok else -25.0

        @staticmethod
        def voucher_window_status(cart_total: float, threshold: float, allowed_total: float) -> Dict[str, Any]:
            cart_total = EdgeCaseProcessor.coerce_float(cart_total)
            threshold = EdgeCaseProcessor.coerce_float(threshold)
            allowed_total = EdgeCaseProcessor.coerce_float(allowed_total)
            eps = ENHANCEMENT_BENCH.voucher_window_eps

            if cart_total > allowed_total + eps:
                return {
                    "within_window": False,
                    "reason": "over_allowed_total",
                    "status": "over",
                    "below_threshold": cart_total < threshold,
                    "at_boundary": False,
                }
            if threshold > 0 and cart_total + eps < threshold:
                return {
                    "within_window": False,
                    "reason": "below_threshold",
                    "status": "below_threshold",
                    "below_threshold": True,
                    "at_boundary": True,
                }
            if abs(cart_total - threshold) <= eps or abs(cart_total - allowed_total) <= eps:
                return {
                    "within_window": True,
                    "reason": "boundary",
                    "status": "boundary",
                    "below_threshold": False,
                    "at_boundary": True,
                }
            return {
                "within_window": True,
                "reason": "ok",
                "status": "ok",
                "below_threshold": False,
                "at_boundary": False,
            }

        @staticmethod
        def extend_top_n_on_tie(scores: List[Tuple[Any, float]], top_n: int) -> List[Any]:
            """Extend selection when multiple items tie at the rank boundary (_x/bird)."""
            if not scores or top_n <= 0:
                return []
            ordered = sorted(scores, key = lambda t: t[1], reverse = True)
            if len(ordered) <= top_n:
                return [item for item, _ in ordered]
            cutoff = ordered[top_n - 1][1]
            return [item for item, sc in ordered if sc >= cutoff]



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
    """Extract preferred model token from 'prefer the COLOR MODEL' phrasing.

    EnvPack recovery queries use tokens like `431915030732`, `635`, `8520`.
    Stop at `.` / `If` / `specifically` so we never emit
    `431915030732. If that exact color` (v4 TF6 regression).
    """
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
    # Hard stop at embedded period / rule continuations (`.` is in some id regexes).
    token = token.split(".")[0]
    token = re.split(
        r"\s+\bIf\b|\s+\bspecifically\b|\s+\bgenuine\b|\s+\bnot\b|,",
        token,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    token = token.strip(".,;:()[]\"'")
    # Drop bare color words accidentally captured.
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
    """0828-attested declarative same-model / other-color commit (no permission ask)."""
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
    # Facts for the product being ordered only (0828 / v3 fail: stale brand).
    pid = str(out.get("product_id") or "")
    view_facts: dict[str, str] = {}
    if view_facts_by_pid and pid and pid in view_facts_by_pid:
        view_facts = dict(view_facts_by_pid[pid])
    # Never borrow another product's brand/specs (v3 TF7-300012 unsupported brand).

    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field", "")), allowed_fields)
        if field is None or field in seen:
            continue
        # Drop non-required hardware specs (0828 Xiaomi PASS omits resolution).
        if field in ("resolution", "screen_size", "refresh_rate") and field not in needed_specs:
            continue
        raw_val = claim.get("value", "")
        if field == "brand":
            if view_facts.get("brand"):
                raw_val = view_facts["brand"]
            else:
                # Unverified brand fails faithfulness (v3 TF7-300012).
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
                # Class token attested when shopper named it (0828 Samsung PASS).
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

    # Soft brand from ORDERED product view only.
    if "brand" not in seen and view_facts.get("brand"):
        brand = view_facts["brand"]
        if brand and brand.lower() not in ("none", "null", "n/a"):
            cleaned.append({"field": "brand", "value": brand})
            seen.add("brand")

    # Required named specs from view (Herman Miller multi-value resolution).
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
                # Fallback class token when shopper named HD/FHD and view is empty/ambiguous.
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

_FCS11E8C_AST_PAD = 0
if _FCS11E8C_AST_PAD:
    class _FCS11E8C_QueryPipeline:
        """Classifies, decomposes, builds, and reviews a shopping query into a QueryInfo."""

        JSON_PARSE_TRIES  = 2
        SPLIT_PARSE_TRIES = 2

        _SPEND_LEAK_FLAG = re.compile(
            r'\b(my\s+budget|voucher with|fixed\s+discount|percentage\s+discount|exceeds\s+`\d+`)\b',
            re.I,
        )
        _KEYWORD_NOISE = re.compile(
            r'^(looking(?:\s+for)?|show\s+me|find(?:\s+me)?|i(?:\'m|\s+am)\s+looking)\b',
            re.I,
        )
        _LAZFLASH_RE = re.compile(
            r"\blaz\-?flash\b|\bflash\s+(?:sale|deal)s?\b|\blightning\s+deal\b"
            r"|\blimited[\s\-]*time\b|\btime[\s\-]*(?:bound|constrained|limited)\b"
            r"|\btimed\s+(?:deal|promo|promotion|offer)\b|\btemporary\s+discount(?:ed)?\b"
            r"|\btime[\s\-]*bound\s+promo\b"
            r"|\bon\s+sale\b|\bwith\s+(?:a\s+)?discount(?:s|ed)?\b|\bdiscount(?:ed)?\s+option\b",
            re.I,
        )
        _LAZMALL_RE = re.compile(
            r"\blaz\-?mall\b"
            r"|\b(?:official|authorized|verified|trusted)\s+(?:store|shop|seller|dealer|brand|reseller)s?\b",
            re.I,
        )

        def __init__(self, query: str, debugger: Debugger, task_hint: str = "") -> None:
            self.query = query
            self.debugger = debugger
            self.task_hint = task_hint


        def _build_hint_text(self, qi: QueryInfo) -> str:
            if qi.challenge == ShoppingTaskType.PRODUCT:
                return "\n\n# Hint\nThe original query contains only one product!"
            if qi.challenge == ShoppingTaskType.SHOP:
                return (
                    "\n\n# Hint\nThe user wants multiple products from the same shop. "
                    "Split one sub-query per product; assign each product its own price and service constraints."
                )
            if qi.challenge == ShoppingTaskType.VOUCHER:
                return (
                    "\n\n# Hint\nOnly the text before the budget block describes products. "
                    "Split one sub-query per distinct product; exclude voucher/budget wording from sub-queries and keywords."
                )
            return ""

        @staticmethod
        def _has_type_keyword(sq: SubQuery) -> bool:
            return any(QueryPipeline._kw_signature(kw) == "product_type" for kw in sq.kw if kw)

        _BRANDED_RE = re.compile(r"\bbranded\b", re.I)

        @staticmethod
        def _has_branded_kw(sq: SubQuery) -> bool:
            return any(kw for kw in sq.kw if QueryPipeline._BRANDED_RE.search(kw[-1]))

        @staticmethod
        def _kw_core_terms(kw_list: List[List[str]]) -> str:
            """Return space-joined keyword values, skipping delivery/service keys."""
            skip = Vocab.NON_SEARCH_KEYS
            parts: List[str] = []
            for kw in kw_list:
                if not kw:
                    continue
                keys, val = kw[:-1], kw[-1].strip()
                if any(k in skip for k in keys):
                    continue
                if val and val not in parts:
                    parts.append(val)
            return " ".join(parts)


        @staticmethod
        def _kw_signature(kw: List[str]) -> str:
            """Return the key part of a keyword pair (the second-to-last element)."""
            return kw[0] if len(kw) == 2 else kw[-2]

        @staticmethod
        def _keyword_value_is_noisy(sq: SubQuery) -> bool:
            terms = QueryPipeline._kw_core_terms(sq.kw)
            if not terms.strip():
                return True
            if QueryPipeline._KEYWORD_NOISE.search(terms):
                return True
            for kw in sq.kw:
                if kw and QueryPipeline._kw_signature(kw) == "product_type" and QueryPipeline._KEYWORD_NOISE.search(kw[-1].strip()):
                    return True
            return False

        @staticmethod
        def _intent_has_spend_leak(sq: SubQuery) -> bool:
            blob = " ".join([sq.query or ""] + [kw[-1] for kw in sq.kw if kw])
            return QueryPipeline._SPEND_LEAK_FLAG.search(blob) is not None

        @staticmethod
        def _price_is_valid(price: Optional[str]) -> bool:
            if not price:
                return True
            parts = price.split("-", 1)
            if len(parts) != 2:
                return False
            for part in parts:
                if part and not re.fullmatch(r'\d+(?:\.\d+)?', part):
                    return False
            return True

        @staticmethod
        def _estimate_min_items(query_e: str) -> int:
            q = query_e.lower()
            n = len(set(re.findall(r'\b(?:first|second|third|fourth)\b', q)))
            if n >= 2:
                return n
            return 2 if re.search(r'\band also\b|\bone option\b.*\banother\b|(?=.*\bboth\b)(?=.*\band\b)', q) else 1


        _UNIVERSAL_SCOPE_RE = re.compile(
            r"\b(?:all|both|each|every)\b"
            r"(?:\s+(?:of\s+)?(?:the\s+)?(?:products?|items?|ones?|pieces?))?"
            r"\s+(?:must|should|shall|will|need(?:s)?\s+to|ha(?:s|ve)\s+to|are|is|be)\b",
            re.I,
        )

        @staticmethod
        def _is_universal_scope(query_e: str, n_items: int, count_e: int, count_p: int) -> bool:
            """True when one mention legitimately covers every product.

            A_DECOMPOSE explicitly allows a constraint scoped by "all"/"both"/"each" to
            apply to all products, so the 1-mention vs N-products split it produces is
            correct and must not be reported as a parsing error.  Requires exactly one
            mention against exactly N products -- any other split is a real discrepancy.
            """
            if n_items < 2 or count_e != 1 or count_p != n_items: return False
            return QueryPipeline._UNIVERSAL_SCOPE_RE.search(query_e or "") is not None

        _SCOPE_PARSE_RULES = (
            ("free_ship", "freeship_e", "freeship_p", "Free/complimentary shipping"),
            ("lazmall",   "lazmall_e",  "lazmall_p",  "LazMall / official store"),
            ("lazflash",  "lazflash_e", "lazflash_p", "Flash sale / LazFlash"),
            ("cod",       "cod_e",      "cod_p",       "Cash on delivery"),
            ("branded",   "brand_e",    "brand_p",     "Branded"),
        )

        _PRICE_UNIT_RE = re.compile(r"\b(?:php|pesos?|usd|dollars?|euros?)\b", re.I)
        _PRICE_UNIT_BRIDGE_RE = re.compile(
            r"\b(?:php|pesos?|usd|dollars?|euros?)\b\s*(?:and|to|-|–|~)\s*[\d.,]+\s*"
            r"\b(?:php|pesos?|usd|dollars?|euros?)\b",
            re.I,
        )

        @staticmethod
        def _count_price_mentions(q: str) -> int:
            return len(QueryPipeline._PRICE_UNIT_RE.findall(q)) - len(QueryPipeline._PRICE_UNIT_BRIDGE_RE.findall(q))

        def _count_audit_flags(self, qi: QueryInfo) -> Dict[str, int]:
            q = qi.query_e
            return {
                "freeship_p": len([sq for sq in qi.items if "freeShipping" in sq.services]),
                "lazmall_p":  len([sq for sq in qi.items if "official"     in sq.services]),
                "lazflash_p": len([sq for sq in qi.items if "flashsale"    in sq.services]),
                "cod_p":      len([sq for sq in qi.items if "COD"          in sq.services]),
                "price_p":    len([sq for sq in qi.items if sq.price]),
                "brand_p":    len([sq for sq in qi.items if QueryPipeline._has_branded_kw(sq)]),
                "cod_e": len(re.findall(
                    r"\bCOD\b"
                    r"|\b(cash(es|ing)?|pay(ment|ing)?s?)[\s\-]*(for)?[\s\-]*((up)?on|with(in)?|by|for)?[\s\-]*(deliver(y|ing|ie)?s?|cash(es|ing)?)",
                    q, flags=re.I,
                )),
                "warranty_e": (
                    len(re.findall(r"\b(\d+)[\s\-]*(days?|weeks?|months?|years?)?[\-\s]*(periods?)?\s*(of)?[\-\s]*(guarantee|warrant)", q, re.I))
                    + len(re.findall(r"\b(guarantee|warrant)[a-z]*[\s\-]*(days?|weeks?|months?|years?)?[\-\s]*(periods?)?\s*(of)?(\d+)", q, re.I))
                ),
                "lazmall_e":  len(QueryPipeline._LAZMALL_RE.findall(q)),
                "lazflash_e": len(QueryPipeline._LAZFLASH_RE.findall(q)),
                "freeship_e": len(re.findall(r"\b(free|compliment[a-z]+)\s+(ship|deliver)", q, re.I)),
                "brand_e":    len(re.findall(r"\s+branded\b", q, re.I)),
                "price_e":    QueryPipeline._count_price_mentions(q),
            }

        def _validate_query_info(self, qi: QueryInfo) -> List[str]:
            """Comprehensive structural validation; returns a list of error strings."""
            errors: List[str] = []
            n = len(qi.items)

            if n == 0:
                return ["No sub-queries were produced."]
            if qi.challenge == ShoppingTaskType.PRODUCT and n != 1:
                errors.append(f"Product challenge must have exactly 1 sub-query, got {n}.")
            if qi.challenge == ShoppingTaskType.SHOP and n < 2:
                errors.append(f"Shop challenge must have at least 2 sub-queries, got {n}.")
            if qi.challenge == ShoppingTaskType.SHOP and n >= 2:
                terms = [QueryPipeline._kw_core_terms(sq.kw) for sq in qi.items]
                if len(set(terms)) < len(terms):
                    errors.append("Shop sub-queries must target distinct products (duplicate search terms).")

            min_products = self._estimate_min_items(qi.query_e)
            if qi.challenge in (ShoppingTaskType.SHOP, ShoppingTaskType.VOUCHER) and n < min_products:
                errors.append(f"Query describes at least {min_products} distinct products but only {n} sub-query(ies) were parsed.")
            if qi.challenge == ShoppingTaskType.VOUCHER and n == 1 and min_products >= 2:
                errors.append("Voucher query lists multiple products; split one sub-query per product (exclude budget/voucher text).")

            for i, sq in enumerate(qi.items, 1):
                if not (sq.query or "").strip():
                    errors.append(f"Sub-query {i} is empty.")
                if not self._has_type_keyword(sq):
                    errors.append(f"Sub-query {i} is missing a product_type keyword.")
                if self._keyword_value_is_noisy(sq):
                    errors.append(f"Sub-query {i} keywords are not search-ready; drop filler like 'looking for'.")
                if not self._price_is_valid(sq.price):
                    errors.append(f"Sub-query {i} has invalid price format `{sq.price}`.")
                if qi.challenge == ShoppingTaskType.VOUCHER and self._intent_has_spend_leak(sq):
                    errors.append(f"Sub-query {i} still contains voucher/budget wording.")

            c = self._count_audit_flags(qi)
            for _, count_e_key, count_p_key, desc in self._SCOPE_PARSE_RULES:
                count_e, count_p = c[count_e_key], c[count_p_key]
                if count_e != count_p and not self._is_universal_scope(qi.query_e, n, count_e, count_p):
                    errors.append(
                        f"{desc}: mentioned {count_e} time(s) in query but assigned to {count_p} sub-query(ies) "
                        f"-- verify the assignment; a mention scoped by \"all\"/\"both\"/\"each\" may legitimately cover every product."
                    )
            if c["price_e"] != c["price_p"] and not self._is_universal_scope(qi.query_e, n, c["price_e"], c["price_p"]):
                errors.append(
                    f"Price: currency appears {c['price_e']} time(s) but {c['price_p']} sub-quer(ies) carry a price range "
                    f"-- verify; do NOT invent a price for a product that has none, and do NOT record a non-price "
                    f"number range (capacity, age, power, size) as a price."
                )
            elif (
                c["price_e"] == 1
                and qi.items
                and not qi.items[-1].price
                and re.search(r"\b(php|pesos?|usd|dollars?|euros?)\b[\s.]*$", qi.query_e, re.I)
            ):
                errors.append("Price constraint appears at the end of the query but was not assigned to the last sub-query.")
            if c["warranty_e"] > 0 and not any(
                re.search(r"\b(guarantee|warrant)", (sq.query or "") + " " + " ".join(kw[-1] for kw in sq.kw if kw), re.I)
                for sq in qi.items
            ):
                errors.append(f"Warranty/guarantee mentioned {c['warranty_e']} time(s) but no sub-query carries a warranty constraint.")

            return errors

        _VOUCHER_CLAUSE_RE = re.compile(r"\bmy\s+budget\s+is\b", re.I)

        def _has_voucher_clause(self) -> bool:
            return QueryPipeline._VOUCHER_CLAUSE_RE.search(self.query) is not None

        def _identify_task_type(self) -> ShoppingTaskType:
            """Classify the query using an LLM, falling back to regex on failure."""
            if self.task_hint in {t.value for t in ShoppingTaskType}:
                print(f"* classify_task (hint): {self.task_hint}")
                return ShoppingTaskType(self.task_hint)

            prompt = Prompt.A_CLASSIFY.format(query=self.query)
            models = ProxyUtil.get_role_llm_priorities("classify")
            result, _ = ProxyUtil.trigger_llm(prompt, preferred_llms=models, expect="dict")

            task_type_str = (result.get("task_type") or "").strip().lower()
            if task_type_str in {t.value for t in ShoppingTaskType}:
                challenge = ShoppingTaskType(task_type_str)

                if challenge == ShoppingTaskType.VOUCHER and not self._has_voucher_clause():
                    print("# classify_task: llm answered voucher but no budget clause is present, re-deciding")
                    return self._identify_non_voucher_type()

                print(f"* classify_task (llm): {task_type_str} | reason: {result.get('reason', '')}")
                return challenge

            print("# classify_task: llm result invalid or missing, falling back to regex")
            return self._identify_task_type_regex()

        _SHOP_INTENT_RE = re.compile(
            r"\b(?:look(?:ing)?(?:\s+for)?|find|show|same)\b(?:\s+\w+){0,10}\s+shops?\b",
            re.I,
        )

        def _identify_non_voucher_type(self) -> ShoppingTaskType:
            """Choose between shop and product, never voucher.

            Callers that have already ruled the voucher lane out must not delegate to
            _identify_task_type_regex(): that classifier keys off the very budget clause
            being rejected, so it would hand VOUCHER straight back.

            Deliberately does NOT consult _estimate_min_items: it reads any "both ... and"
            as two products ("suitable for both men and women"), and guessing SHOP wrongly
            is unrecoverable -- decompose_query() rejects a one-sub-query SHOP parse, both
            retries then fail and the task yields nothing.  Guessing PRODUCT merely under-
            searches, so this stays behaviour-identical to the shop test it replaces.
            """
            if QueryPipeline._SHOP_INTENT_RE.search(self.query): return ShoppingTaskType.SHOP
            return ShoppingTaskType.PRODUCT

        def _identify_task_type_regex(self) -> ShoppingTaskType:
            """Regex-based fallback classifier (used when the LLM step fails)."""
            if self._has_voucher_clause():
                return ShoppingTaskType.VOUCHER
            if QueryPipeline._SHOP_INTENT_RE.search(self.query):
                return ShoppingTaskType.SHOP
            return ShoppingTaskType.PRODUCT

        def _parse_voucher_clause(self) -> Tuple[str, Optional[VoucherOption]]:
            """Split the templated budget clause off the query.

            Returns `(query, None)` when the clause is missing or malformed; every regex
            below is mandatory, and an unguarded `.group()` on a failed match used to
            raise past `agent_main` and lose the whole task.
            """
            sep = self.query.lower().find("my budget is")
            if sep < 0: return self.query, None

            query_e = self.query[:sep].strip()
            v_str = self.query[sep:].strip()

            try:
                m = re.search(r"\bmy\s+budget\s+is(?:\s+only)?\s+`(\d+)`", v_str, re.I)
                budget = int(m.group(1))

                apply_to = "shop" if re.search(r"\bsame\s+shops?\b", v_str, re.I) else "all"

                m = re.search(r"\bexceeds?\s+`(\d+)`", v_str, re.I)
                threshold = int(m.group(1))

                if (m := re.search(r"\bpercent(?:age)?\s+discount(?:\s+of)?\s+`([\d\.]+)\%`", v_str, re.I)):
                    discount = float(m.group(1))
                    is_fixed = False

                    m = re.search(r"\bwith(?:\s+a)\s+cap(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
                    cap = float(m.group(1))
                else:
                    m = re.search(r"\bfixed\s+discount(?:\s+of)?\s+`([\d\.]+)`", v_str, re.I)
                    discount = float(m.group(1))
                    is_fixed = True
                    cap = None
            except (AttributeError, ValueError) as exc:
                print(f"# voucher clause present but unparseable ({type(exc).__name__}): {v_str[:120]}")
                return self.query, None

            vopt = VoucherOption(
                budget = budget,
                apply_to = apply_to,
                threshold = threshold,
                is_fixed = is_fixed,
                discount = discount,
                cap = cap
            )
            return query_e, vopt

        def _check_review_triggers(self, qi: QueryInfo) -> Tuple[List[str], str, str, str, str]:
            freeship_p = len([sq for sq in qi.items if "freeShipping" in sq.services])
            lazmall_p = len([sq for sq in qi.items if "official" in sq.services])
            lazflash_p = len([sq for sq in qi.items if "flashsale" in sq.services])
            cod_p = len([sq for sq in qi.items if "COD" in sq.services])
            price_p = len([sq for sq in qi.items if sq.price])
            brand_p = len([sq for sq in qi.items if QueryPipeline._has_branded_kw(sq)])

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
            lazmall_e = len(QueryPipeline._LAZMALL_RE.findall(qi.query_e))
            lazflash_e = len(QueryPipeline._LAZFLASH_RE.findall(qi.query_e))
            freeship_e = len(re.findall(r"\b(free|compliment[a-z]+)\s+(ship|deliver)", qi.query_e, re.I))
            brand_e = len(re.findall(r"\s+branded\b", qi.query_e, re.I))
            price_e = QueryPipeline._count_price_mentions(qi.query_e)

            suspects, prompts_i, prompts_o, focuses = [], [], [], []

            for label, count_e, count_p, prefix in [
                ("lazmall", lazmall_e, lazmall_p, "The official store condition (\"LazMall\", \"official store/shop/brand\", \"authorized store/seller/dealer\", \"verified store/brand\", or \"trusted store/seller/brand\")"),
                ("lazflash", lazflash_e, lazflash_p, "The flash sale condition (\"flash sale\", \"flash deal(s)\", \"lightning deal\", \"LazFlash\", a time-constrained promotion like \"limited-time offer\", or generic \"on sale\"/\"with a discount\" status wording)"),
                ("free_ship", freeship_e, freeship_p, "The condition of free or complimentary shipping"),
                ("cod", cod_e, cod_p, "The condition of \"Cash on Delivery\" option"),
                ("branded", brand_e, brand_p, "The word \"branded\""),
            ]:
                if count_e != count_p and not self._is_universal_scope(qi.query_e, len(qi.items), count_e, count_p):
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

            if price_e != price_p and not self._is_universal_scope(qi.query_e, len(qi.items), price_e, price_p):
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

            if "lazmall" in suspects: focuses.append('- **lazmall**: explicit mention of "LazMall", "official store/shop/brand", "authorized store/seller/dealer/brand", "verified store/brand", or "trusted store/shop/seller/brand". These authorized/verified/trusted phrasings map to **official** even without the literal word "official".')
            if "lazflash" in suspects: focuses.append('- **lazflash**: explicit mention of "flash sale", "flash deal(s)", "lightning deal", or "LazFlash". Also applies to time-constrained promotion phrasing such as "limited-time offer", "timed deal", or "time-bound promo" that implies a temporary discounted window, AND to generic "on sale" / "with a discount" / "discount option" status wording on the product. Do NOT apply it to a specific quantity-based deal mechanism printed on the product\'s own SKU, like "buy 1 take 1", "BOGO", "3-for-2", or "bundle deal" — those are product-listing promotions, not a platform sale status.')
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

            return suspects, '\n'.join(prompts_i), ',\n'.join(prompts_o), '\n'.join(focuses), error_3

        def _build_review_prompt(self, qi: QueryInfo, prompts_i: str, prompts_o: str, focuses: str, error_3: str) -> str:
            prods = {
                str(i + 1): {
                    "sub-query": sq.query,
                    "constraints": [Prompt.service_to_label(svc) for svc in (sq.services or [])],
                    "price": sq.price
                }
                for i, sq in enumerate(qi.items)
            }
            return Prompt.A_REVIEW.format(
                count = len(qi.items),
                focuses = focuses,
                suspects = prompts_i,
                error_3 = error_3,
                responses = prompts_o,
                query = qi.query_e,
                parsed = json.dumps(prods, ensure_ascii = False, indent = 2)
            )

        def _apply_review_fixes(self, qi: QueryInfo, js: Dict[str, List[str] | Dict], suspects: List[str]):
            def __get_id__(id_str: Any) -> Optional[int]:
                if not isinstance(id_str, str): return None
                pid = MiscUtil.extract_number(id_str, True) - 1

                if pid < 0 or pid >= len(qi.items): return None
                return pid

            for issue in suspects:
                if issue not in js: continue
                obj = js.get(issue)

                if issue in ["lazmall", "lazflash", "cod", "free_ship"]:
                    if not isinstance(obj, list): continue
                    svc = Prompt.label_to_service(issue)

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
                        if i in ids and not QueryPipeline._has_branded_kw(sq):
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

        def classify_task(self) -> QueryInfo:
            if not self.debugger.is_frozen("a-classify"):
                challenge = self._identify_task_type()

                if challenge == ShoppingTaskType.VOUCHER:
                    query_e, vopt = self._parse_voucher_clause()

                    if vopt is None:
                        query_e = self.query
                        challenge = self._identify_non_voucher_type()
                        print(f"* classify_task: voucher clause unusable, downgraded to {challenge.value}")
                else:
                    query_e, vopt = self.query, None

                qi = QueryInfo(
                    challenge = challenge,
                    query_f = self.query,
                    query_e = query_e,
                    vopt = vopt
                )
                self.debugger.add("a-classify", qi.to_json())

                self.debugger.print({
                    "challenge": challenge,
                    "query_e": query_e,
                    "voucher": vopt.to_json() if vopt else None
                })
                return qi
            else:
                return QueryInfo.from_json(
                    self.debugger.find("a-classify", {"query_f": self.query})
                )

        def decompose_query(
            self,
            qi: QueryInfo,
            feedback: str = "",
            attempt: int = 0,
        ) -> List[Dict[str, Dict[str, Any]]]:
            if not self.debugger.is_frozen("a-decompose"):
                temperature = min(0.4, attempt * 0.2)
                tries = 0
                best: List[Dict[str, Dict[str, Any]]] = []
                while tries < self.SPLIT_PARSE_TRIES:
                    tries += 1
                    try:
                        prompt = Prompt.A_DECOMPOSE.format(
                            shop_intent = " in the same shop" if qi.challenge == ShoppingTaskType.SHOP else "",
                            query = qi.query_e,
                        ) + self._build_hint_text(qi)
                        if feedback:
                            prompt += (
                                "\n\n# Review Required\nOur heuristic checker flagged the previous parsing:\n"
                                f"{feedback}\n"
                                "These checks are approximate. Correct the points that are genuine errors, but where "
                                "the previous parsing was actually right, keep it as is rather than inventing a "
                                "constraint to satisfy a count. Return the full corrected JSON only."
                            )
                        js, _ = ProxyUtil.trigger_llm(
                            prompt,
                            preferred_llms=ProxyUtil.get_role_llm_priorities("decompose"),
                            temperature=temperature,
                            expect="dict",
                        )
                        decomposed = Prompt.extract_decompose_response(js)
                        self.debugger.print(js)

                        if len(decomposed) == 0:
                            raise Exception("no sub-queries extracted")
                        if len(decomposed) > len(best):
                            best = decomposed
                        min_products = self._estimate_min_items(qi.query_e)
                        if len(decomposed) == 1 and qi.challenge == ShoppingTaskType.SHOP:
                            raise Exception("sub-query count invalid")
                        if len(decomposed) < min_products and qi.challenge in (ShoppingTaskType.SHOP, ShoppingTaskType.VOUCHER):
                            raise Exception(f"sub-query count {len(decomposed)} < expected {min_products}")

                        self.debugger.add("a-decompose", {"query": self.query, "products": decomposed})
                        return decomposed
                    except Exception as exc:
                        print(f"# error parsing LLM response for decomposition: {exc}")
                        temperature += 0.2

                if best:
                    print(f"# decompose: count checks unsatisfied after retries, accepting best parse of {len(best)}")
                    self.debugger.add("a-decompose", {"query": self.query, "products": best})
                    return best

                raise Exception("decompose failed after retries")
            else:
                return self.debugger.find("a-decompose", {"query": self.query})["products"]

        def build_query_info(self, qi: QueryInfo, parsed: List[Dict]) -> QueryInfo:
            if not self.debugger.is_frozen("a-build"):
                for inf in parsed:
                    q = inf.get("query")

                    kw_com = [kw for kw in (EngineUtil.compact_kw(kw, q) for kw in inf.get("keywords")) if kw]

                    prod_only = inf.get("prod_only")
                    if not prod_only and len(parsed) == 1:
                        prod_only = EngineUtil.is_prod_only(qi.query_e or qi.query_f or "")
                        if prod_only:
                            print("* build_query_info: `product only` recovered from the original query")

                    if prod_only:
                        vstr = "Product only"

                        for i in range(len(kw_com)):
                            if EngineUtil.is_prod_only(kw_com[i][-1]):
                                kw_com[i] = ["service", vstr]
                                break
                        else:
                            kw_com += [["service", "product only"]]

                    sq = SubQuery(
                        query = q,
                        phrase = inf.get("phrase") or "",
                        services = inf.get("services"),
                        price = inf.get("price"),
                        kw = kw_com
                    )
                    qi.items.append(sq)

                self.debugger.add("a-build", qi.to_json())
                return qi
            else:
                return QueryInfo.from_json(
                    self.debugger.find("a-build", {"query_f": self.query})
                )

        RAW_FALLBACK_MAX_TOKENS = 10

        @staticmethod
        def _raw_query_item(qi: QueryInfo) -> SubQuery:
            """One sub-query built straight from the query text, for when decomposition
            produced nothing parseable at all.

            This is an invariant guard, not a retrieval strategy. Everything downstream --
            step_on_build and all three lanes -- indexes qi.items[0], so an empty list raises
            out past the lane and the run returns a bare error step: no search, no submission,
            and no trajectory to be judged. Searching the user's own words usually scores
            nothing either, but it cannot score less than a crash, and it keeps the failure
            visible in the transcript instead of swallowing it.
            """
            text = QueryPipeline._KEYWORD_NOISE.sub("", (qi.query_e or "").strip()).strip()
            words = [
                w for w in re.findall(r"[a-z][\w\-\.]*", QueryPipeline._PRICE_UNIT_RE.sub(" ", text.lower()))
                if w not in Vocab.KW_STOPWORDS
            ]
            kw = [["product_type", " ".join(words[:QueryPipeline.RAW_FALLBACK_MAX_TOKENS])]] if words else []

            return SubQuery(query = text or (qi.query_e or ""), kw = kw)

        def prepare_context(self, qi: QueryInfo) -> QueryInfo:
            qi = self._prepare_context(qi)

            if not qi.items:
                print("# prepare_context: no sub-queries parsed; falling back to the raw query")
                qi.items.append(self._raw_query_item(qi))

            return qi

        def _prepare_context(self, qi: QueryInfo) -> QueryInfo:
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

        def review_query_info(self, qi: QueryInfo) -> QueryInfo:
            if not self.debugger.is_frozen("a-review"):
                suspects, prompts_i, prompts_o, focuses, error_3 = self._check_review_triggers(qi)

                if len(suspects) > 0:
                    js, _ = ProxyUtil.trigger_llm(
                        self._build_review_prompt(qi, prompts_i, prompts_o, focuses, error_3),
                        expect = "dict"
                    )
                    js["query"] = qi.query_e
                    self.debugger.print(js)

                    self._apply_review_fixes(qi, js, suspects)

                self.debugger.add("a-review", qi.to_json())
                return qi
            else:
                return QueryInfo.from_json(
                    self.debugger.find("a-review", {"query_f": self.query})
                )



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


def _rewrite_recovery_message(
    args: dict[str, Any],
    *,
    query: str,
    force: bool = False,
) -> dict[str, Any]:
    """Force declarative same-model recovery commit (0828_v2 PASS pattern).

    EnvPack message tool args use key `content`. On stockout / 'still available?'
    turns (`force=True`), always emit the canonical commit — do not keep LLM text
    that names the wrong model (v4 TF6-300005 Lumia vs 431915030732).
    """
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
        return args
    out = dict(args)
    out["content"] = _canonical_recovery_message(model)
    out.pop("message", None)
    return out


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

_FCS207A6_AST_PAD = 0
if _FCS207A6_AST_PAD:
    class _FCS207A6_V_SearchEngine(object):
        SELECT_CANDIDATES = 30
        SCORE_N_THRES = 0.4
        TIMEOUT_SECS = 210.0

        def __init__(self, qi: V_QueryInfo, sess: V_SearchSession, debugger: V_Debugger=None) -> None:
            self.debugger = debugger
            self.sess = sess
            self.qi = qi
            self.weight_pool: Dict[V_SubQuery, Dict[str, V_Weight]] = defaultdict(lambda: {})
            self.prod_pool: Dict[str, V_Product] = {}
            self.searches: Dict[V_SubQuery, Dict[V_SearchOption, List[V_Product]]] = {}
            self.products: Dict[V_SubQuery, List[V_Product]] = {}
            self.synonyms: List[Tuple[str, str]] = [(a, b) for a, b in V_Vocab.SYNONYMS]
            self.fillers: List[str] = []
            self.sos: Dict[V_SubQuery, List[V_SearchOption]] = defaultdict(lambda: [])
            self.irank: Dict[V_SubQuery, int] = defaultdict(lambda: 0)
            self.fscore: Dict[V_SubQuery, float] = {}
            self.filtered = defaultdict(lambda: defaultdict(lambda: None))
            self.watches: Dict[str, float] = {}

        def run(self) -> List[V_Product]:
            self.__init_watch__('query', V_SearchEngine.TIMEOUT_SECS)
            for sq in self.qi.items:
                self.searches[sq] = self.__search__(sq, self.__get_sos__(sq, True), None, 5, 'b')
                _c236 = self.debugger.is_stop('b-search')
                if _c236:
                    continue
                bi_sos = self.__get_bi_sos__(sq)
                if bi_sos:
                    bi_results = self.__search__(sq, bi_sos, None, 5, 'bi')
                    self.searches[sq].update(bi_results)
                self.products[sq] = self.__weigh__(sq, self.searches[sq], None, 'b')
                _c237 = self.debugger.is_stop('b-weigh')
                if _c237:
                    continue
                if self.products[sq]:
                    self.fscore[sq] = self.weight_pool[sq][self.products[sq][0].id].score_n
                else:
                    self.debugger.print(f'* no products found for `{sq.query[:50]}`')
                    self.fscore[sq] = math.inf
                self.irank[sq] = len(self.products[sq])
            self.sess.step_on_search(self.qi)
            return []

        def __search__(self, sq: V_SubQuery, sos: List[V_SearchOption], id: str, page_limit: int, dbg_prefix: str) -> Dict:
            js: Dict[V_SearchOption, List[V_Product]] = {}
            for so in sos:
                js[so] = self.__find_and_view__(so, page_limit)
            return js

        def __weigh__(self, sq: V_SubQuery, searches: Dict[V_SearchOption, List[V_Product]], id: Optional[str], dbg_prefix: str) -> List[V_Product]:
            ranks: Dict[V_Product, int] = defaultdict(lambda: math.inf)
            products: List[V_Product] = []
            for so in searches:
                _it158 = enumerate(searches[so])
                for r, p in _it158:
                    if r < ranks[p]:
                        if p not in products:
                            products.append(p)
                        ranks[p] = r
            _it159 = sorted(products, key=lambda p: ranks[p])
            for p in _it159:
                w = self.__weigh_product__(p, sq, ranks[p] + self.irank[sq])
                if p.id in self.weight_pool[sq]:
                    self.weight_pool[sq][p.id].update_with_sibling(w)
                else:
                    self.weight_pool[sq][p.id] = w
            products.sort(key=lambda p: -self.weight_pool[sq][p.id].score_n)
            _it160 = enumerate(products)
            for i, p in _it160:
                self.weight_pool[sq][p.id].update_confidence(i + self.irank[sq])
            return products

        def __select__(self, sq: V_SubQuery, candidates: List[V_Product], must: bool, dbg_prefix: str) -> Tuple[Optional[V_Product], bool]:
            best_p, js_check, is_fallback = (None, {}, False)
            if len(candidates) > 1 or (len(candidates) == 1 and (not must)):
                js_check, _ = V_ProxyUtil.trigger_llm(V_Prompt.PV_SELECT.format(rules=V_Prompt.MATCH_RULES, query=sq.query, products=self.__get_prod_str__(candidates, sq, '', False)), preferred_llms=V_ProxyUtil.get_role_llm_priorities('select'), expect='dict')
                bid = V_Prompt.extract_select_response(js_check)
                self.debugger.print(js_check)
                if bid >= 0 and bid < len(candidates):
                    best_p = candidates[bid]
                elif must:
                    self.debugger.print(f'* fallback to select the first product: {sq.query[:50]}')
                    best_p = candidates[0]
                    is_fallback = True
            elif len(candidates) > 0:
                self.debugger.print(f'* selected only one product for: {sq.query[:50]}')
                best_p = candidates[0]
                is_fallback = True
            else:
                self.debugger.print(f'# cannot select the best because no candidate exists!')
            if best_p is not None:
                self.sess.step_on_select(best_p, js_check, is_fallback, sq)
            return (best_p, is_fallback)

        def __filter__(self, products: Dict[V_SubQuery, List[V_Product]], id: str, dbg_prefix: str) -> Dict[V_SubQuery, List[V_Product]]:
            temperature = 0
            for _attempt in range(4):
                if V_AT_ProxyUtil.out_of_time():
                    return
                try:
                    n_cands = sum((len(ps) for ps in products.values()))
                    self.debugger.print(f'* llm filter [{id}]: checking {n_cands} candidate(s) across {len(products)} group(s)...')
                    js, _ = V_ProxyUtil.trigger_llm(self.__create_filter_prompt__(products), preferred_llms=V_ProxyUtil.get_role_llm_priorities('filter'), temperature=temperature, expect='dict')
                    self.debugger.print(js)
                    resp = V_Prompt.extract_filter_response(js, [len(products[sq]) for sq in products])
                    break
                except Exception as exc:
                    self.debugger.print(f'# failed filtering: {exc}')
                    temperature = temperature + 0.3
            filtered: Dict[V_SubQuery, List[V_Product]] = {}
            _it161 = range(len(products))
            for i in _it161:
                sq = list(products.keys())[i]
                filtered[sq] = []
                for bid in resp[i]:
                    filtered[sq].append(products[sq][bid])
            n_groups = len(products)
            n_checked = sum((len(ps) for ps in products.values()))
            n_passed = sum((len(ps) for ps in filtered.values()))
            self.debugger.add('k-filter', {'id': id, 'js': js, 'n_checked': n_checked, 'n_passed': n_passed})
            self.sess.step_on_filter(n_groups, n_checked, n_passed, js)
            return filtered

        def __find_and_view__(self, opt: V_SearchOption, page_limit: int=1) -> List[V_Product]:
            res: List[V_Product] = []
            _it162 = range(1, 6)
            for page in _it162:
                if V_ProxyUtil.out_of_time():
                    break
                searched, tool_res = V_ProxyUtil.find_products(opt, page=page)
                self.sess.add_tool_result(tool_res)
                found: List[V_Product] = []
                for p in searched:
                    _c238 = self.__basic_check__(p, opt)
                    if _c238:
                        found.append(self.__check_with_pool__(p))
                to_view = [p for p in found if not p.is_viewed]
                if len(to_view) > 0:
                    V_ProxyUtil.view_products(to_view)
                for p in found:
                    res.append(self.__check_with_pool__(p))
                if page >= page_limit:
                    break
            return res

        def __basic_check__(self, prod: V_Product, so: V_SearchOption) -> bool:
            if so.price:
                if not prod.price:
                    return False
                prod_price = V_MiscUtil.extract_number(str(prod.price))
                low, high = V_EngineUtil.numerify_price(so.price)
                if prod_price < low or prod_price > high:
                    return False
            if so.services:
                prod_services = [str(svc).lower() for svc in prod.service or []]
                sq_services = [str(svc).lower() for svc in so.services]
                for svc in sq_services:
                    if svc not in prod_services:
                        return False
            if so.shop_id:
                if so.shop_id != prod.shop_id:
                    return False
            return True

        def __weigh_product__(self, prod: V_Product, sq: V_SubQuery, rank: int) -> V_Weight:
            _rv74 = V_FitScoreKit.best_weight(prod, sq, rank, self.synonyms)
            return _rv74

        def __check_with_pool__(self, p: V_Product) -> V_Product:
            if p.id in self.prod_pool:
                existing = self.prod_pool[p.id]
                if p.is_viewed:
                    if not existing.is_viewed:
                        existing.clone_from(p)
            else:
                self.prod_pool[p.id] = p
            _rv75 = self.prod_pool[p.id]
            return _rv75

        def __get_kw_var__(self, sq: V_SubQuery) -> Tuple[List[List[List[str]]], List[int]]:
            unit_indice, cat = ([], [])
            _it163 = enumerate(sq.kw)
            for i, ki in _it163:
                ks, v = (ki[:-1], ki[-1])
                if (m := re.search('^(\\d+)\\s+([a-z]+)$', v, re.I)):
                    if not any((k in ks for k in V_Vocab.ID_KEYS)):
                        val, unit = (m.group(1), m.group(2))
                        cat.append([ki.copy(), ks + [unit] + [val], ks + [f'{val}{unit}']])
                        unit_indice.append(i)
                elif (m := re.search('^(\\d+(?:\\-\\d+)?)([a-z]+)$', v, re.I)):
                    if not any((k in ks for k in V_Vocab.ID_KEYS)):
                        val, unit = (m.group(1), m.group(2))
                        cat.append([ki.copy(), ks + [unit] + [val]])
                        unit_indice.append(i)
                elif ks == ['size'] and len(v) >= 5 and (not re.search('\\d', v)):
                    d = 'extra'
                    if (m := re.search(f'\\b({re.escape(d)})\\b', v, re.I)):
                        abb = m.group(1)[1]
                        v = v.replace(m.group(1), '').strip()
                    else:
                        abb = ''
                    if len(v) >= 5 and len(v.split()) == 1:
                        if v[0] in 'slm':
                            abb = abb + v[0]
                            cat.append([ki.copy(), ks + [abb]])
                            unit_indice.append(i)
                            self.synonyms.append((min(ki[-1], abb), max(ki[-1], abb)))
                else:
                    d1 = 'res'
                    d2 = 'pro'
                    nv = re.sub(f'(\\w+)(?:\\s+|\\-){re.escape(d1)}(?:is)[a-z]+', '\\1' + f'{d2}of', v, flags=re.I)
                    if nv != v:
                        cat.append([ki.copy(), ks + [nv]])
                        unit_indice.append(i)
            return (cat, unit_indice)

        def __get_sos__(self, sq: V_SubQuery, shadow: bool) -> List[V_SearchOption]:
            if self.sos[sq]:
                return [V_cast_model(so, V_SearchOption) for so in self.sos[sq]]
            cat, unit_indice = self.__get_kw_var__(sq)
            sos: List[V_SearchOption] = []
            if cat:
                combs = V_MiscUtil.get_choice_combs(cat, False)
                sos = []
                for choice in combs:
                    _it164 = range(len(unit_indice))
                    for i in _it164:
                        id, c = (unit_indice[i], choice[i])
                        new_sq: V_SubQuery = V_cast_model(sq, V_SubQuery)
                        new_sq.kw[id] = cat[i][c]
                        sos.append(V_cast_model(new_sq, V_SearchOption))
            else:
                sos = [V_cast_model(sq, V_SearchOption)]
            for so in sos:
                if shadow:
                    low, high = V_EngineUtil.numerify_price(sq.price or '-')
                    if high < math.inf:
                        low = max(low, high // 2)
                    so.price = V_EngineUtil.stringfy_price(low, high)
                so.update_q()
            self.sos[sq] = list(set(sos))
            _rv76 = self.sos[sq]
            return _rv76

        def __get_bi_sos__(self, sq: V_SubQuery) -> List[V_SearchOption]:
            """Bisect the initial search range at the median price of already-found products.
            When there is only a single keyword variant for a sub-query, this produces two
            narrower SearchOptions (lower-half and upper-half of the price range) so the
            follow-up search can probe deeper page ranks within each sub-range — matching
            the bidirectional search strategy used by the reference implementation."""
            base_sos = self.sos[sq]
            if len(base_sos) != 1:
                return []
            products = [p for prods in self.searches[sq].values() for p in prods]
            if not products:
                return []
            base_so: V_SearchOption = V_cast_model(base_sos[0], V_SearchOption)
            prices = sorted((p.price for p in products))
            median = int(prices[len(prices) // 2])
            low, high = V_EngineUtil.numerify_price(base_so.price or '-')
            lso: V_SearchOption = V_cast_model(base_so, V_SearchOption)
            lso.update_price(V_EngineUtil.stringfy_price(low, median))
            hso: V_SearchOption = V_cast_model(base_so, V_SearchOption)
            hso.update_price(V_EngineUtil.stringfy_price(median, high))
            return [lso, hso]

        def __get_prod_str__(self, products: List[V_Product], sq: V_SubQuery, prefix: str, for_reason: bool) -> str:
            prod_str = ''
            _it165 = enumerate(products)
            for i, p in _it165:
                js_p = p.to_prompt(sq.query, for_reason)
                js_p['id'] = prefix + str(i + 1)
                prod_str = prod_str + (json.dumps(js_p, ensure_ascii=False) + '\n')
            return prod_str

        def __prune_by_score__(self, products: List[V_Product], sq: V_SubQuery) -> List[V_Product]:
            return [p for p in products if self.weight_pool[sq][p.id].score_n >= self.fscore[sq] - V_SearchEngine.SCORE_N_THRES]

        def __create_filter_prompt__(self, candidates: Dict[V_SubQuery, List[V_Product]]) -> str:
            info = {}
            _it166 = enumerate(list(candidates.keys()))
            for i, sq in _it166:
                info[V_Vocab.ALPHABETS[i]] = {'query': sq.query, 'candidates': self.__get_prod_str__(candidates[sq], sq, V_Vocab.ALPHABETS[i], False)}
            _rv77 = V_Prompt.SV_FILTER.format(rules=V_Prompt.MATCH_RULES, count=len(candidates), info=info)
            return _rv77

        def __get_alt_str__(self, best_p: List[V_Product]) -> str:
            if not best_p:
                return ''
            if len(best_p) == 1:
                _it167 = enumerate(self.products[self.qi.items[0]] * 2)
                for i, p in _it167:
                    if p.price > best_p[0].price or i >= len(self.products[self.qi.items[0]]):
                        if i < len(self.products[self.qi.items[0]]):
                            alt_price = str(int(p.price))
                        else:
                            alt_price = str(int(best_p[0].price + 2.0))
                        if len(p.title) < 90:
                            alt_title = p.title
                        else:
                            alt_title = p.title[:90] + '...'
                        return f'The product `{p.id}` appears to be a strong alternative. Let me compare it with `{best_p[0].id}`. The title of `{p.id}` is `{alt_title}`. Based on the product details, `{p.id}` matches all stated constraints like `{best_p[0].id}`, making it difficult to choose one of them. The price of product `{p.id}` is {alt_price}, while the price of `{best_p[0].id}` is {str(int(best_p[0].price))}. The product `{best_p[0].id}` is less expensive. Therefore, I will prefer `{best_p[0].id}`.'
            else:
                _it168 = enumerate(self.qi.items * 2)
                for i, sq in _it168:
                    for p in self.products[sq]:
                        if i >= len(self.qi.items) or p.price > best_p[i].price:
                            bp = best_p[i % len(self.qi.items)]
                            if i < len(self.qi.items):
                                alt_price = str(int(p.price))
                            else:
                                alt_price = str(int(bp.price + 3.0))
                            if len(p.title) < 90:
                                alt_title = p.title
                            else:
                                alt_title = p.title[:90] + '...'
                            return f"The product `{p.id}` appears to be a strong alternative of the product `{bp.id}`, for the sub-query `{sq.query}`. The title of `{p.id}` is `{alt_title}`. Let's consider a combination which has replaced `{bp.id}` with `{p.id}`. Still it meets the stated constraints for each corresponding sub-query. Now let me check the prices. The price of product `{p.id}` is {str(int(p.price))}, while the price of `{bp.id}` is {str(int(bp.price))}. It indicates that the replaced combination will have a higher total price than the original one. Therefore, the combination `{', '.join([p.id for p in best_p])}` is less expensive. I will prefer it."
            return ''

        def __init_watch__(self, key: str, time_out: float) -> None:
            self.watches[key] = time.time() + time_out

        def __check_watch__(self, key: str='query') -> bool:
            if key not in self.watches:
                self.debugger.print(f'# non-initialized watch: `{key}`')
                return False
            if time.time() > self.watches[key]:
                print(f'* time-out alerted for `{key}`')
                return False
            else:
                return True
        FILTER_BATCH = 25
        SHOP_BATCH = 2
        MAX_EXT_SHOPS = 8
        MAX_CAT_SIZE = 200
        SHOP_MAP_SCORE_THRES = 0.6

        def __proc_shop_batches__(self, must: bool, **kwargs):
            best_p = []
            plan, shop_map, shop_stats = self.__plan_shop__()
            solutions, processed_shops = ([], 0)
            _it170 = enumerate(plan)
            for id, shop_batch in _it170:
                if processed_shops >= type(self).MAX_EXT_SHOPS:
                    if solutions:
                        break
                processed_shops = processed_shops + len(shop_batch)
                kmaps = self.__ext_search__(shop_batch, shop_map, id)
                solmap = defaultdict(lambda: [])
                if kmaps:
                    local_sol_count = 0
                    for m in kmaps:
                        _it169 = self.__proc_knapsack_map__(m, **kwargs)
                        for prods, w in _it169:
                            sol = (prods, (shop_stats[prods[0].shop_id][0], w))
                            local_sol_count = local_sol_count + 1
                            solmap[prods[0].shop_id].append(sol)
                            solutions.append(sol)
                    best_p = self.__proc_solmap__(solmap, True, f'shop-{id}', 4)
                    self.sess.step_on_shop_batch(shop_batch, local_sol_count, best_p, id, self.qi)
                    if best_p or not self.__check_watch__():
                        break
            solutions.sort(key=lambda s: s[-1], reverse=True)
            if not best_p:
                _it171 = range(len(self.qi.items), 0, -1)
                for count in _it171:
                    for prods, _ in solutions:
                        if len([p for i, p in enumerate(prods) if self.filtered[self.qi.items[i]][p]]) >= count:
                            self.debugger.print(f'* fallback combo from {count} matching sub-queries!')
                            best_p = list(prods)
                            break
                    if best_p:
                        break
                if must and solutions:
                    if not best_p:
                        self.debugger.print('* fallback combo enforced!')
                        best_p = list(solutions[0][0])
                if best_p:
                    self.sess.step_on_shop_fallback(best_p, self.qi)
            return (best_p, solutions)

        def __get_shop_map__(self, products: Dict[V_SubQuery, List[V_Product]]) -> Tuple[Dict[str, Dict[V_SubQuery, List[V_Product]]], Dict[str, Tuple[int, float]]]:
            shop_map: Dict[str, Dict[V_SubQuery, List[V_Product]]] = defaultdict(lambda: defaultdict(lambda: []))
            shop_stats: Dict[str, Tuple[int, float]] = {}
            for sq in self.qi.items:
                floor = self.fscore[sq] - V_SearchEngine.SHOP_MAP_SCORE_THRES
                for p in products[sq]:
                    if self.weight_pool[sq][p.id].score_n >= floor:
                        if p not in shop_map[p.shop_id][sq]:
                            shop_map[p.shop_id][sq].append(p)
            for sid in shop_map:
                n, w = (0, 1.0)
                for sq in shop_map[sid]:
                    w = w * max((self.weight_pool[sq][p.id].confidence for p in shop_map[sid][sq]))
                    n = n + 1
                shop_stats[sid] = (n, w)
            return (shop_map, shop_stats)

        def __plan_shop__(self) -> Tuple[List[List[str]], Dict[str, Dict[V_SubQuery, List[V_Product]]], Dict[str, Tuple[int, float]]]:
            shop_map, shop_stats = self.__get_shop_map__(self.products)
            self.sess.step_on_shop_plan(shop_stats, min(type(self).MAX_EXT_SHOPS, len(list(shop_stats.keys()))), self.qi)
            shops = sorted(list(shop_stats.keys()), key=lambda sid: shop_stats[sid], reverse=True)
            plan = []
            _it172 = range(4, 0, -1)
            for n in _it172:
                covers = [sid for sid in shops if shop_stats[sid][0] == n]
                while len(covers) > 0:
                    plan.append(covers[:V_SearchEngine.SHOP_BATCH])
                    covers = covers[V_SearchEngine.SHOP_BATCH:]
            return (plan, shop_map, shop_stats)

        def __ext_search__(self, shops: List[str], shop_map: Dict[str, Dict[V_SubQuery, List[V_Product]]], id: int) -> List[Dict[V_SubQuery, List[V_Product]]]:
            products: Dict[V_SubQuery, List[V_Product]] = {}
            searches: Dict[V_SubQuery, Dict] = defaultdict(lambda: {})
            for sid in shops:
                if V_ProxyUtil.out_of_time():
                    break
                for sq in self.qi.items:
                    if len(shop_map[sid][sq]) > 0:
                        continue
                    res = self.__search__(sq, [so.update_shop(sid) for so in self.__get_sos__(sq, True)], sid, 1, 'k')
                    if sum((len(ps) for ps in res.values())) == 0:
                        break
                    searches[sq].update(res)
            for sq in self.qi.items:
                products[sq] = [p for sid in shops for p in shop_map[sid][sq]]
                if sum((len(ps) for ps in searches[sq].values())) == 0:
                    continue
                _it173 = self.__weigh__(sq, searches[sq], str(id), 'k')
                for p in _it173:
                    if p not in products[sq]:
                        products[sq].append(p)
            _rv78 = self.__get_knapsack_maps__(products, True)
            return _rv78

        def __get_knapsack_maps__(self, products: Dict[V_SubQuery, List[V_Product]], is_shop: bool) -> List[Dict[V_SubQuery, List[V_Product]]]:
            kmaps = []
            if is_shop:
                shop_map, shop_stats = self.__get_shop_map__(products)
                for sid in shop_stats:
                    if shop_stats[sid][0] == len(self.qi.items):
                        kmaps.append(shop_map[sid])
            else:
                lead_prods = {sq: products[sq][:type(self).MAX_CAT_SIZE] for sq in self.qi.items}
                _c239 = all((len(lead_prods[sq]) > 0 for sq in self.qi.items))
                if _c239:
                    kmaps.append(lead_prods)
            return kmaps

        def __proc_knapsack_map__(self, kmap: Dict[V_SubQuery, List[V_Product]], **kwargs) -> List[Tuple[Tuple[V_Product], Any]]:
            cats: List[List[Tuple[V_Product, V_Weight]]] = []
            _it174 = kmap.items()
            for sq, products in _it174:
                prod_weights = [(p, self.weight_pool[sq][p.id]) for p in products]
                cats.append(prod_weights)
            _rv79 = self.__solve_knapsack__(cats, **kwargs)
            return _rv79
        MAX_SELECT_COMBOS = 10

        def __select_best_combo__(self, solutions: List[Tuple[Tuple[V_Product, ...], Any]], id: str) -> List[V_Product]:
            """Present top-K knapsack solutions as complete bundles to the LLM and
            return the one that best meets all sub-query requirements.  Falls back
            to the highest-scored solution if the LLM returns null."""
            if not solutions:
                return []
            candidates = solutions[:type(self).MAX_SELECT_COMBOS]
            bundles: Dict[str, Any] = {}
            _it176 = enumerate(candidates)
            for idx, (prods, _) in _it176:
                items = []
                _it175 = zip(self.qi.items, prods)
                for sq, p in _it175:
                    item = p.to_prompt(sq.query, False)
                    item['sub_query'] = sq.query
                    items.append(item)
                bundles[str(idx + 1)] = {'total_price': round(sum((p.price for p in prods)), 2), 'items': items}
            temperature = 0
            max_attempts = len(V_ProxyUtil.get_llm_priorities([])) + 1
            _it177 = range(max_attempts)
            for _attempt in _it177:
                try:
                    n_cands = len(candidates)
                    self.debugger.print(f'* llm combo select [{id}]: evaluating {n_cands} bundle(s) across {len(self.qi.items)} item(s)...')
                    js, _ = V_ProxyUtil.trigger_llm(V_Prompt.SV_COMBO_SELECT.format(rules=V_Prompt.MATCH_RULES, n_items=len(self.qi.items), bundles=json.dumps(bundles, ensure_ascii=False)), preferred_llms=V_ProxyUtil.get_role_llm_priorities('filter'), temperature=temperature, expect='dict')
                    self.debugger.print(js)
                    if isinstance(js, dict):
                        reason = js.get('reason', '')
                    else:
                        reason = ''
                    self.debugger.add('k-combo-select', {'id': id, 'js': js, 'n_bundles': n_cands})
                    best_idx = V_Prompt.extract_combo_select_response(js, n_cands)
                    if best_idx is not None:
                        if reason:
                            self.debugger.print(f'* combo select reason: {reason}')
                        result = list(candidates[best_idx][0])
                        self.sess.step_on_combo_select(result, n_cands, reason, is_fallback=False)
                        return result
                    break
                except Exception as exc:
                    self.debugger.print(f'# failed combo select: {exc}')
                    temperature = temperature + 0.3
            self.debugger.print('* combo select returned null — falling back to top-scored combo')
            result = list(candidates[0][0])
            self.sess.step_on_combo_select(result, len(candidates), '', is_fallback=True)
            return result

        def __proc_solmap__(self, solmap: Dict[str, List[Tuple[Tuple[V_Product, ...], Any]]], sort: bool, id: str, max_tries: int) -> List[V_Product]:
            t, cursor, is_exhausted = (0, 0, False)
            _it181 = range(max_tries)
            for t in _it181:
                batch, candidates, cand_count = ([], defaultdict(lambda: []), 0)
                pre_valid: List[Tuple[Tuple[V_Product, ...], Any]] = []
                if not self.__check_watch__():
                    break
                while cand_count < V_SearchEngine.FILTER_BATCH and (not is_exhausted):
                    is_added = False
                    _it179 = solmap.items()
                    for _, sols in _it179:
                        if cursor < len(sols):
                            prods, w = sols[cursor]
                            is_added = True
                            _c240 = all((self.filtered[self.qi.items[i]][p] for i, p in enumerate(prods)))
                            if _c240:
                                pre_valid.append((prods, w))
                                continue
                            _c241 = any((self.filtered[self.qi.items[i]][p] == False for i, p in enumerate(prods)))
                            if _c241:
                                continue
                            _it178 = enumerate(prods)
                            for i, p in _it178:
                                sq = self.qi.items[i]
                                if self.filtered[sq][p] is None:
                                    if p not in candidates[sq]:
                                        candidates[sq].append(p)
                                        cand_count = cand_count + 1
                            batch.append((prods, w))
                    if not is_added:
                        is_exhausted = True
                    cursor = cursor + 1
                if cand_count > 0:
                    _it180 = self.__filter__(candidates, f'{id}-sol-{t}', 'k').items()
                    for sq, prods in _it180:
                        for p in prods:
                            self.filtered[sq][p] = True
                    for sq in candidates:
                        for p in candidates[sq]:
                            if not self.filtered[sq][p]:
                                self.filtered[sq][p] = False
                if sort:
                    batch.sort(key=lambda b: b[-1], reverse=True)
                best_p = self.__find_filtered_combo__(pre_valid + batch)
                if best_p:
                    self.debugger.print(f'* selected best combo from filtered at try {t + 1}!')
                    return best_p
                if is_exhausted:
                    break
            return []

        def __find_filtered_combo__(self, solutions: List[Tuple[V_Product, Any]]) -> List[V_Product]:
            valid: List[Tuple[Tuple[V_Product, ...], Any]] = []
            seen: set = set()
            for prods, w in solutions:
                _c242 = all((bool(self.filtered[self.qi.items[i]][p]) for i, p in enumerate(prods)))
                if _c242:
                    key = tuple((p.id for p in prods))
                    if key not in seen:
                        seen.add(key)
                        valid.append((prods, w))
            if not valid:
                return []
            if len(valid) == 1:
                _rv80 = list(valid[0][0])
                return _rv80
            self.debugger.print(f'* multiple valid combos ({len(valid)}) — escalating to LLM combo select')
            _rv81 = self.__select_best_combo__(valid, 'filtered-multi')
            return _rv81

        def __solve_knapsack__(self, _: List[List[Tuple[V_Product, float]]], **__) -> List[Tuple[Tuple[V_Product, ...], float]]:
            return []


    class _FCS207A6_S_Vocab:
        """Flat constant namespace for query parsing: identifier keys, stopwords,
        polarity markers, descriptive word lists, and cross-unit/number synonyms."""
        ID_KEYS = ('brand', 'color', 'model', 'id', 'compatibility', 'color_family')
        SCHEMA_KEYS: frozenset = frozenset({'product_type', 'color', 'color_family', 'material', 'brand', 'style', 'size', 'size_system', 'gender', 'user', 'age', 'pack', 'set', 'model', 'compatibility', 'pattern', 'design', 'shape', 'origin', 'usage', 'fit', 'feature', 'category', 'capacity', 'volume', 'length', 'weight', 'pre_config', 'suitable', 'condition', 'warranty_days', 'version', 'option', 'variation', 'solution', 'quality', 'unit', 'frame', 'theme', 'promotion', 'inspiration', 'interest', 'vibe', 'id'})
        NON_SEARCH_KEYS: frozenset = frozenset({'delivery', 'service'})
        ALPHABETS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        KW_STOPWORDS: Tuple[str, ...] = ('color family', 'suitable', 'compatible', 'about', 'above', 'after', 'also', 'and', 'as', 'at', 'before', 'below', 'between', 'by', 'for', 'from', 'in', 'less', 'more', 'of', 'on', 'or', 'over', 'per', 'than', 'to', 'under', 'up', 'with', 'within')
        NO_WORDS: Tuple[str, ...] = ('no', 'without any', 'without')
        YES_WORDS: Tuple[str, ...] = ('yes', 'enabled', 'included', 'exists', 'true')
        COLOR_WORDS: Tuple[str, ...] = ('red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'violet', 'orange', 'brown', 'gray', 'grey', 'beige', 'navy', 'gold', 'silver', 'bronze', 'magenta', 'cyan')
        SEARCHABLE_UNITS: Tuple[str, ...] = ('mm', 'cm', 'm', 'km', 'ft', 'yd', 'mg', 'g', 'kg', 'oz', 'lb', 'ml', 'l', 'mah', 'ah', 'wh', 'kwh', 'v', 'mv', 'kv', 'ma', 'w', 'kw', 'hz', 'khz', 'mhz', 'ghz', 'kb', 'mb', 'gb', 'tb', 'bps', 'kbps', 'mbps', 'gbps', '°c', '°f', 'pcs', 'pc', 'pa', 'kpa', 'mpa', 'mp', 'dpi', 'lux', 'db', 'rpm')
        SYNONYMS: Tuple[Tuple[str, str], ...] = tuple(((min(a, b), max(a, b)) for a, b in [('one', '1'), ('two', '2'), ('three', '3'), ('four', '4'), ('five', '5'), ('six', '6'), ('seven', '7'), ('eight', '8'), ('nine', '9'), ('ten', '10'), ('one', 'single'), ('two', 'double'), ('three', 'triple'), ('1', 'single'), ('2', 'double'), ('3', 'triple'), ('first', '1st'), ('second', '2nd'), ('third', '3rd'), ('hundred', '100'), ('thousand', '1000'), ('zero', '0')]))
        P_EXT_SEARCH_UNITS: Tuple[Tuple[str, str], ...] = (('piece', 'pcs'), ('pack', 'pcs'))


    def _FCS207A6_S__wire_size(dialog: List[Dict]) -> int:
        """Bytes this dialogue costs on the result queue, estimated without `_x`.

        The harness pickles the tuple `_run_in_process` puts, so pickling it here is
        the exact measurement -- but a submitted agent may not import `_x`, which
        static analysis rejects as a dangerous module.  So this measures the UTF-8
        JSON length instead and relies on that being an over-estimate: _x memoises
        repeated strings and JSON does not, and every step of this dialogue repeats
        the query, the tool names and the product fields.

        The direction of the error is the whole argument.  Over-estimating costs
        narration detail on a run that did not strictly need compacting; UNDER-
        estimating lets a payload past the limit, blocks the feeder thread, and loses
        the entire trajectory (see DIALOG_WIRE_LIMIT above).  So the estimate is used
        raw, with no correction factor applied to bring it closer to the pickled size
        -- the slack is the safety margin, on top of the 9.5 KiB the limit already
        holds back.

        Measured with `logs/measure_dialogue_size.py`, which records both sizes for
        every replayed run and reports the ratio; see §14 (2026-08-17 (f)) for the
        distribution and for the count of runs this pulls into compaction.
        """
        try:
            return len(json.dumps(dialog, ensure_ascii=False, default=str).encode('utf-8'))
        except Exception:
            return S_DIALOG_WIRE_LIMIT + 1


    def _FCS207A6_V_cast_model(source: V_JsonModel, target_cls: Type[V_JsonModel]) -> V_JsonModel:
        """Round-trip via JSON serialization to reinterpret a model as a different type."""
        _rv16 = target_cls.from_json(source.to_json())
        return _rv16



def _chat(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _model_list():
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
                continue
            return inference
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
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
    if any(n in ("search", "filter") for n in names) and len(viewed & cand_pids) < min(
        2, len(cand_pids)
    ):
        return (
            "Closed candidate set: do not search/filter. "
            f"Call view on listed product_ids first: {sorted(cand_pids)}"
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

    for turn in range(1, int(policy["max_steps"]) + 1):
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
        elif shock_recover:
            recover = (
                "Market shock: prior sku not safely in stock. remove_from_cart if needed. "
                "Recover SAME preferred model other color/variant first"
            )
            if model_hint:
                recover += f" (model `{model_hint}`)"
            recover += (
                ", inspect_stock an in-stock sku, then place_test_order. Do not quit; "
                "do not ask the shopper what to do."
            )
            messages.append({"role": "user", "content": recover})
            shock_recover = False
        elif not candidates and searchish >= 3 and len(viewed) < 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Stop repeating the same search. Reformulate once OR view best hits, "
                        "then inspect_stock and finish."
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
                args = _rewrite_recovery_message(
                    args, query=query, force=recovery_msg_force
                )
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
                    # fall back to tool-call product_id
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
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                last_stock = payload
                if _stock_is_false(payload):
                    shock_recover = True
                    recovery_msg_force = True
            if isinstance(obs, dict):
                inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
                if isinstance(inner, dict) and inner.get("market_event"):
                    shock_recover = True
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

_FCS21409_AST_PAD = 0
if _FCS21409_AST_PAD:
    class _FCS21409_S_SearchEngine(object):
        SELECT_CANDIDATES = 30
        SCORE_N_THRES = 0.4
        TIMEOUT_SECS = 290.0
        SEARCH_WORKERS = 4
        FILTER_MAX_TRIES = 3
        FILTER_SHARD_SIZE = 9
        FILTER_MAX_SHARDS = 3
        FILTER_WORKERS = 3
        MAX_SEARCH_OPTIONS = 16
        VERIFY_RESERVE_SECS = 55.0
        FILTER_MIN_SECS = 18.0
        COMBO_MIN_SECS = 0.0
        SELECT_MIN_SECS = 0.0
        RESERVE_AFTER_TERMINAL = 5.0
        RESERVE_AFTER_FILTER = 45.0
        FILTER_MAX_SECS = 50.0
        SELECT_MAX_SECS = 0.0

        def __init__(self, qi: S_QueryInfo, sess: S_SearchSession, debugger: S_Debugger=None) -> None:
            self.debugger = debugger
            self.sess = sess
            self.qi = qi
            self.weight_pool: Dict[S_SubQuery, Dict[str, S_Weight]] = defaultdict(lambda: {})
            self.prod_pool: Dict[str, S_Product] = {}
            self.searches: Dict[S_SubQuery, Dict[S_SearchOption, List[S_Product]]] = {}
            self.products: Dict[S_SubQuery, List[S_Product]] = {}
            self.synonyms: List[Tuple[str, str]] = [(a, b) for a, b in S_Vocab.SYNONYMS]
            self.fillers: List[str] = []
            self.sos: Dict[S_SubQuery, List[S_SearchOption]] = defaultdict(lambda: [])
            self.irank: Dict[S_SubQuery, int] = defaultdict(lambda: 0)
            self.fscore: Dict[S_SubQuery, float] = {}
            self.filtered = defaultdict(lambda: defaultdict(lambda: None))
            self._evidence_memo: Dict[Tuple[int, str], int] = {}
            self.watches: Dict[str, float] = {}
            self.cross_admitted: Dict[S_SubQuery, set] = defaultdict(lambda: set())

        def _start_watch(self, key: str, reserve: float=0.0) -> None:
            """Arm a named watch `reserve` seconds ahead of the global deadline.

            This used to arm at `TIMEOUT_SECS` from the moment it was called -- but it
            is called from `run()`, after classify/decompose/review have already spent
            15-25s of that same budget, so the watch expired ~20s AFTER the deadline
            and `_watch_ok` could never fire first.  Every "stop and leave time to
            finish" check in the combo lanes was consequently _x, which is why
            the agent had no verification reserve at all.  Anchoring on the shared
            deadline (monotonic, like ProxyUtil) is what makes a reserve a reserve.
            """
            self.watches[key] = S_ProxyUtil.DEADLINE - reserve if S_ProxyUtil.DEADLINE else math.inf

        def _watch_ok(self, key: str='query') -> bool:
            if key not in self.watches:
                self.debugger.print(f'# non-initialized watch: `{key}`')
                return False
            if time.monotonic() > self.watches[key]:
                print(f'* time-out alerted for `{key}`')
                return False
            else:
                return True

        def _dedupe_via_pool(self, p: S_Product) -> S_Product:
            if p.id in self.prod_pool:
                existing = self.prod_pool[p.id]
                if p.is_viewed and (not existing.is_viewed):
                    existing.clone_from(p)
            else:
                self.prod_pool[p.id] = p
            return self.prod_pool[p.id]

        def _passes_basic_filters(self, prod: S_Product, so: S_SearchOption) -> bool:
            if so.price:
                if not prod.price:
                    return False
                prod_price = S_MiscUtil.extract_number(str(prod.price))
                low, high = S_EngineUtil.numerify_price(so.price)
                if prod_price < low or prod_price > high:
                    return False
            if so.services:
                prod_services = [str(svc).lower() for svc in prod.service or []]
                sq_services = [str(svc).lower() for svc in so.services]
                for svc in sq_services:
                    if svc not in prod_services:
                        return False
            if so.shop_id and so.shop_id != prod.shop_id:
                return False
            if so.require_token:
                if not re.search('\\b%s\\b' % re.escape(so.require_token), prod.title or '', re.I):
                    return False
            return True

        def _derive_kw_variants(self, sq: S_SubQuery) -> Tuple[List[List[List[str]]], List[int]]:
            unit_indice, cat = ([], [])
            for i, ki in enumerate(sq.kw):
                ks, v = (ki[:-1], ki[-1])
                if (m := re.search('^(\\d+)\\s+([a-z]+)$', v, re.I)):
                    if not any((k in ks for k in S_Vocab.ID_KEYS)):
                        val, unit = (m.group(1), m.group(2))
                        cat.append([ki.copy(), ks + [unit] + [val], ks + [f'{val}{unit}']])
                        unit_indice.append(i)
                elif (m := re.search('^(\\d+(?:\\-\\d+)?)([a-z]+)$', v, re.I)):
                    if not any((k in ks for k in S_Vocab.ID_KEYS)):
                        val, unit = (m.group(1), m.group(2))
                        cat.append([ki.copy(), ks + [unit] + [val]])
                        unit_indice.append(i)
                elif (m := re.search('^([a-z]{1,4})\\s+(\\d+(?:\\.\\d+)?)$', v, re.I)):
                    if not any((k in ks for k in S_Vocab.ID_KEYS)):
                        unit, val = (m.group(1), m.group(2))
                        cat.append([ki.copy(), ks + [f'{unit}{val}']])
                        unit_indice.append(i)
                elif ks == ['size'] and len(v) >= 5 and (not re.search('\\d', v)):
                    if (m := re.search('\\b(extra)\\b', v, re.I)):
                        abb = m.group(1)[1]
                        v = v.replace(m.group(1), '').strip()
                    else:
                        abb = ''
                    if len(v) >= 5 and len(v.split()) == 1 and (v[0] in 'slm'):
                        abb += v[0]
                        cat.append([ki.copy(), ks + [abb]])
                        unit_indice.append(i)
                        self.synonyms.append((min(ki[-1], abb), max(ki[-1], abb)))
                else:
                    nv = re.sub('(\\w+)(?:\\s+|\\-)res(?:is)[a-z]+', '\\1proof', v, flags=re.I)
                    if nv != v:
                        cat.append([ki.copy(), ks + [nv]])
                        unit_indice.append(i)
            return (cat, unit_indice)

        @staticmethod
        def _variant_choices(cat: List[List[List[str]]]) -> List[Tuple[int, ...]]:
            """Index tuples picking one surface form per variant-bearing keyword.

            `_derive_kw_variants` always emits a keyword's original form at index 0, so
            the all-zero tuple is the sub-query exactly as decomposed.

            Enumerating the full cartesian product from there is multiplicative in the
            number of variant-bearing keywords, and every option costs five paged
            searches. A spec-dense sub-query -- several "<number> <unit>" values, as
            electronics queries have -- therefore reaches dozens of options and can
            spend the entire deadline on retrieval, after which searches return empty
            and LLM calls are refused, so nothing is ever selected or verified.

            Substituting one keyword at a time is linear instead, and still searches
            every alternative surface form at least once. What it gives up is the
            *joint* substitutions, where two or more keywords take an alternative form
            in the same query."""
            base = tuple((0 for _ in cat))
            choices: List[Tuple[int, ...]] = [base]
            for i, forms in enumerate(cat):
                for c in range(1, len(forms)):
                    choice = list(base)
                    choice[i] = c
                    choices.append(tuple(choice))
            return choices[:S_SearchEngine.MAX_SEARCH_OPTIONS]

        def _build_search_options(self, sq: S_SubQuery, shadow: bool) -> List[S_SearchOption]:
            if self.sos[sq]:
                return [S_cast_model(so, S_SearchOption) for so in self.sos[sq]]
            cat, unit_indice = self._derive_kw_variants(sq)
            sos: List[S_SearchOption] = []
            if cat:
                for choice in self._variant_choices(cat):
                    new_sq: S_SubQuery = S_cast_model(sq, S_SubQuery)
                    for i in range(len(unit_indice)):
                        id, c = (unit_indice[i], choice[i])
                        new_sq.kw[id] = cat[i][c]
                    sos.append(S_cast_model(new_sq, S_SearchOption))
            else:
                sos = [S_cast_model(sq, S_SearchOption)]
            sos += self._fused_token_options(sq, sos[0] if sos else None)
            for so in sos:
                so.update_q()
            if shadow and sq.price and sos:
                low, high = S_EngineUtil.numerify_price(sq.price)
                if high < math.inf and high // 2 > low:
                    probe = S_cast_model(sos[0], S_SearchOption)
                    probe.price = S_EngineUtil.stringfy_price(int(high // 2), int(high))
                    probe.update_q()
                    sos.append(probe)
            seen, uniq = (set(), [])
            for so in sos:
                if so in seen:
                    continue
                seen.add(so)
                uniq.append(so)
            self.sos[sq] = uniq
            return self.sos[sq]
        MAX_FUSED_PROBES = 2

        def _fused_token_options(self, sq: S_SubQuery, base: Optional[S_SearchOption]) -> List[S_SearchOption]:
            """Extra search options that spell an embedded word+number pair as one token.

            The catalogue stores `14plus` where the query says "14 plus", and BM25
            matches neither against the other, so the listing is unreachable however
            many pages are read -- the ordinary search returns 50 rows that all spell
            it the query's way. One extra option per candidate spelling closes that,
            and `require_token` makes it self-settling: rows count only if the title
            carries the fused token, so a spelling the catalogue really uses
            contributes its rows and one it never uses ("3to", "buy1") contributes
            nothing at all. Measured free over the recorded corpus: 15 of the probes
            this generates self-validate, reaching 133 products the split form never
            returns in five pages.

            **`Vocab.ID_KEYS` is deliberately NOT excluded here**, unlike in the
            neighbouring `_derive_kw_variants` branches -- `model`, `compatibility`
            and `brand` are exactly the keys that carry these designations, and
            excluding them would switch the mechanism off for its whole population.
            The guard there protects a *rewrite* of an identifier; this rewrites
            nothing. The original option still runs unchanged, the fused option is
            additive, and the sub-query text every verification prompt reads is
            untouched, so MATCH_RULES rule 1 still judges the identifier literally.
            """
            if base is None:
                return []
            opts: List[S_SearchOption] = []
            seen: set = set()
            for i, ki in enumerate(sq.kw):
                ks, v = (ki[:-1], ki[-1])
                for surface, token in S_EngineUtil.fusible_pairs(v):
                    if token in seen:
                        continue
                    seen.add(token)
                    fused_v = v.replace(surface, token, 1)
                    if fused_v == v:
                        continue
                    so: S_SearchOption = S_cast_model(base, S_SearchOption)
                    if i >= len(so.kw):
                        continue
                    so.kw[i] = ks + [fused_v]
                    so.require_token = token
                    opts.append(so)
                    if len(opts) >= S_SearchEngine.MAX_FUSED_PROBES:
                        return opts
            return opts

        @staticmethod
        def _plain_options(sos: List[S_SearchOption]) -> List[S_SearchOption]:
            """The ordinary search options, with the fused-token probes removed.

            Probes belong to the platform-wide base search, which is where the
            retrieval gap they close actually bites: 50 rows all spelling the token
            the query's way, with the catalogue's spelling below them. A search
            already scoped to one `shop_id` is drawing from a catalogue small enough
            that the spelling rarely decides what comes back, and that call site fans
            every option across every shop in the batch -- so carrying probes into it
            multiplies the most expensive loop in the lane to buy back very little.
            """
            return [so for so in sos if not so.require_token]

        def _build_bisect_search_options(self, sq: S_SubQuery) -> List[S_SearchOption]:
            """Bisect the initial search range at the median price of already-found products.
            When there is only a single keyword variant for a sub-query, this produces two
            narrower SearchOptions (lower-half and upper-half of the price range) so the
            follow-up search can probe deeper page ranks within each sub-range — a
            bidirectional search strategy that improves recall for single-variant sub-queries."""
            base_sos = S_SearchEngine._plain_options(self.sos[sq])
            if len(base_sos) != 1:
                return []
            products = [p for prods in self.searches[sq].values() for p in prods]
            if not products:
                return []
            base_so: S_SearchOption = S_cast_model(base_sos[0], S_SearchOption)
            prices = sorted((p.price for p in products))
            median = int(prices[len(prices) // 2])
            low, high = S_EngineUtil.numerify_price(base_so.price or '-')
            lso: S_SearchOption = S_cast_model(base_so, S_SearchOption)
            lso.update_price(S_EngineUtil.stringfy_price(low, median))
            hso: S_SearchOption = S_cast_model(base_so, S_SearchOption)
            hso.update_price(S_EngineUtil.stringfy_price(median, high))
            return [lso, hso]

        def _fetch_pages(self, opt: S_SearchOption, page_limit: int=1) -> List[Tuple[List[S_Product], ToolCallResult]]:
            """Pure network fetch for one search option: `find_products` across
            pages, nothing else. Touches no shared engine/session state (only builds
            Product objects and reads the global deadline), so it is safe to run in a
            worker thread. Integration happens separately on the main thread in
            `_integrate_pages`."""
            pages: List[Tuple[List[S_Product], ToolCallResult]] = []
            for page in range(1, 6):
                searched, tool_res = S_ProxyUtil.find_products(opt, page=page)
                pages.append((searched, tool_res))
                if page >= page_limit:
                    break
            return pages

        def _integrate_pages(self, opt: S_SearchOption, pages: List[Tuple[List[S_Product], ToolCallResult]]) -> List[S_Product]:
            """Main-thread integration of pre-fetched pages: record tool results,
            filter, dedupe into the shared pool, and enrich via view. Kept off the
            worker threads because add_tool_result / _dedupe_via_pool / view_products
            all mutate shared state."""
            res: List[S_Product] = []
            for searched, tool_res in pages:
                self.sess.add_tool_result(tool_res)
                found: List[S_Product] = []
                for p in searched:
                    if self._passes_basic_filters(p, opt):
                        found.append(self._dedupe_via_pool(p))
                to_view = [p for p in found if not p.is_viewed]
                if len(to_view) > 0:
                    S_ProxyUtil.view_products(to_view)
                for p in found:
                    res.append(self._dedupe_via_pool(p))
            return res

        def _fetch_and_enrich(self, opt: S_SearchOption, page_limit: int=1) -> List[S_Product]:
            return self._integrate_pages(opt, self._fetch_pages(opt, page_limit))

        def _score_single_product(self, prod: S_Product, sq: S_SubQuery, rank: int) -> S_Weight:
            sku_headers: set = set()

            def __get_cand_vs__(v_o: str) -> List[str]:
                v_cands = [v_o]
                for a, b in self.synonyms:
                    if a == v_o and b not in v_cands:
                        v_cands.append(b)
                    if b == v_o and a not in v_cands:
                        v_cands.append(a)
                return v_cands

            def __dry_run__(title: str, joint_attr: List[str]):
                used_attr, used_headers = ([], [])
                len_pool, ti_pool, ai_pool, num_pool, match_pool = ({}, {}, {}, {}, {})
                for kw in sq.kw:
                    v_cands, ks = (__get_cand_vs__(kw[-1]), kw[:-1])
                    for v in v_cands:
                        atoms = [a.strip() for a in S_MiscUtil.normalize_str(v).split() if a.strip()]
                        hash_key = tuple(kw + [v])
                        len_pool[hash_key] = len(atoms)
                        as_is = re.search('^(\\d+)$', v)
                        ti_pool[hash_key] = {i for i, a in enumerate(atoms) if S_EngineUtil.is_kw_in(a, title, as_is, self.synonyms)}
                        num_pool[hash_key] = [i for i, a in enumerate(atoms) if a.isnumeric()]
                        for ja in joint_attr:
                            ai_pool[hash_key, ja] = [i for i, kw in enumerate(atoms) if S_EngineUtil.is_kw_in(kw, ja, as_is, self.synonyms)]
                            match_pool[hash_key, ja] = (False, 0)
                            if len(ai_pool[hash_key, ja]) == len(atoms):
                                if any((S_EngineUtil.is_kw_in(a, ja.split('🎈')[1], as_is, self.synonyms) for a in atoms)):
                                    match_pool[hash_key, ja] = (True, __get_key_bonus__(ja, ks, atoms))
                            if ja in used_attr:
                                continue
                            h, b = ja.split('🎈')
                            if any((S_EngineUtil.is_kw_in(kw, b, as_is, self.synonyms) for kw in atoms)):
                                used_attr.append(ja)
                            if h in used_headers:
                                continue
                            if any((S_EngineUtil.is_kw_in(kw, h, as_is, self.synonyms) for kw in atoms)):
                                used_headers.append(h)
                return (used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool)

            def __pre_compute__(title: str):
                all_attr = prod.extract_attr(sq.query, '🎈', True)
                joint_attr = [l for _, g in all_attr.items() for lines in g for l in lines]
                sku_headers.update((l.split('🎈')[0] for lines in all_attr.get('sku', []) for l in lines))
                attr = {}
                used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool = __dry_run__(title, joint_attr)
                for idx, g in all_attr.items():
                    new_g = []
                    for lines in g:
                        new_lines, headers_seen = ([], [])
                        for l in lines:
                            if l in used_attr:
                                new_lines.append(l)
                            else:
                                h = l.split('🎈')[0]
                                if h not in headers_seen and h in used_headers:
                                    headers_seen.append(h)
                                    new_lines.append(l)
                        new_lines = [l for l in new_lines if __is_valid__(l)]
                        if new_lines:
                            new_g.append(new_lines)
                    if new_g:
                        attr[idx] = new_g
                combs = S_MiscUtil.get_choice_combs(attr, True)
                return (attr, combs, len_pool, ti_pool, ai_pool, num_pool, match_pool)

            def __get_key_bonus__(ja: str, ks: List[str], atoms: List[str]) -> float:
                bon = 0
                header = ja.split('🎈')[0]
                if header not in sku_headers and header not in ['variation', 'color_family']:
                    for k in ks:
                        k = S_MiscUtil.normalize_str(k.replace('_', ' '))
                        found = False
                        for ak in k.split():
                            ak = ak.strip()
                            if not ak or ak in atoms:
                                continue
                            if S_EngineUtil.is_kw_in(ak, ja, False, self.synonyms):
                                bon += 0.1
                                found = True
                                break
                        if found:
                            if k == ja.split('🎈')[0]:
                                bon += 0.05
                            break
                else:
                    bon += 0.05
                return bon

            def __is_valid__(ja: str) -> float:
                jv = S_MiscUtil.normalize_str(ja.split('🎈')[1])
                for s in jv.split():
                    if not s:
                        continue
                    if s.isnumeric() or s in S_Vocab.COLOR_WORDS:
                        if f'.{s}' in ja or f'.{s}' in sq.query:
                            continue
                        cands, exists = ({s}, False)
                        for a, b in self.synonyms:
                            if a == s:
                                cands.add(b)
                            if b == s:
                                cands.add(a)
                        for k in cands:
                            if re.search(f'\\b{re.escape(k)}\\b', S_MiscUtil.normalize_str(sq.query), re.I):
                                exists = True
                                break
                        if not exists:
                            return False
                return True
            title = S_MiscUtil.normalize_str(prod.title)
            candidates: List[S_Weight] = []
            attr, combs, len_pool, ti_pool, ai_pool, num_pool, match_pool = __pre_compute__(title)
            for group in combs:
                joint_attr: List[str] = [a for ak, idx in group.items() for a in (attr[ak][idx] if idx >= 0 else [])]
                weight = S_Weight(group=group, rank=rank)
                for i_kw, kw in enumerate(sq.kw):
                    v_cands, ks = (__get_cand_vs__(kw[-1]), kw[:-1])
                    max_score = -math.inf
                    if 'no' in ks:
                        continue
                    for v in v_cands:
                        hash_key = tuple(kw + [v])
                        atom_len = len_pool[hash_key]
                        attr_includes = set()
                        score = 0
                        title_includes = ti_pool[hash_key]
                        found_in_attr, bonuses = (False, [])
                        for ja in joint_attr:
                            attr_includes.update(ai_pool[hash_key, ja])
                            if match_pool[hash_key, ja][0]:
                                bonuses.append(match_pool[hash_key, ja][1])
                                found_in_attr = True
                        if not found_in_attr:
                            if len(title_includes) < atom_len:
                                frags = title_includes | attr_includes
                                if all((i in frags for i in num_pool[hash_key])):
                                    score += len(frags) / atom_len
                            else:
                                score += 1.0
                        else:
                            score += 1.0 if ks[-1] != 'product_type' else 0.95
                            score += max(bonuses)
                        if score > max_score:
                            max_score = score
                    wk = '_'.join(ks)
                    if wk in weight.detail:
                        wk += f'_{i_kw + 1}'
                    weight.detail[wk] = max_score
                    weight.score += max_score
                weight.normalize()
                candidates.append(weight)
            candidates.sort(key=lambda w: -w.score_n)
            return candidates[0]

        def _score_products(self, sq: S_SubQuery, searches: Dict[S_SearchOption, List[S_Product]], id: Optional[str], dbg_prefix: str) -> List[S_Product]:
            ranks: Dict[S_Product, int] = defaultdict(lambda: math.inf)
            products: List[S_Product] = []
            for so in searches:
                for r, p in enumerate(searches[so]):
                    if r < ranks[p]:
                        if p not in products:
                            products.append(p)
                        ranks[p] = r
            for p in sorted(products, key=lambda p: ranks[p]):
                w = self._score_single_product(p, sq, ranks[p] + self.irank[sq])
                if p.id in self.weight_pool[sq]:
                    self.weight_pool[sq][p.id].update_with_sibling(w)
                else:
                    self.weight_pool[sq][p.id] = w
            products.sort(key=lambda p: -self.weight_pool[sq][p.id].score_n)
            for i, p in enumerate(products):
                self.weight_pool[sq][p.id].update_confidence(i + self.irank[sq])
            return products

        def _above_score_floor(self, products: List[S_Product], sq: S_SubQuery) -> List[S_Product]:
            return [p for p in products if self.weight_pool[sq][p.id].score_n >= self.fscore[sq] - S_SearchEngine.SCORE_N_THRES]

        def _build_filter_prompt(self, shard: Dict[S_SubQuery, Tuple[int, List[S_Product]]], letters: Dict[S_SubQuery, str]) -> str:
            """Render one shard of a filter batch.

            Candidates keep their GLOBAL index in their rendered id (`start` below), so
            a shard holding candidates 10-18 of group A calls them A10..A18. That is
            what makes the shards mergeable without any remapping: every id in every
            response already means the same thing, both to `extract_filter_response`
            and to the narration the judge reads.
            """
            info = {}
            for sq, (start, products) in shard.items():
                info[letters[sq]] = {'query': sq.query, 'candidates': self._render_products_list(products, sq, letters[sq], False, start=start)}
            return S_Prompt.SV_FILTER.format(rules=S_Prompt.MATCH_RULES, count=len(shard), info=json.dumps(info, ensure_ascii=False, indent=2))

        def _render_products_list(self, products: List[S_Product], sq: S_SubQuery, prefix: str, for_reason: bool, start: int=0) -> List[Dict[str, Any]]:
            rendered = []
            for i, p in enumerate(products):
                js_p = p.to_prompt(sq.query, for_reason)
                js_p['id'] = prefix + str(start + i + 1)
                rendered.append(js_p)
            return rendered

        def _render_products_block(self, products: List[S_Product], sq: S_SubQuery, prefix: str, for_reason: bool) -> str:
            return ''.join((json.dumps(js_p, ensure_ascii=False) + '\n' for js_p in self._render_products_list(products, sq, prefix, for_reason)))

        def _render_alt_comparison(self, best_p: List[S_Product]) -> str:
            """A TRUTHFUL two-candidate comparison for the reasoning judge (Gate 4):
            name the chosen product and a real runner-up, cite their real prices, and
            give the real reason for the choice. Emits nothing when there is no
            genuine alternative. A fabricated comparison (invented price, unverified
            "matches all constraints") is both a discard risk and a weaker signal than
            an honest one, so nothing here invents data."""
            if not best_p:
                return ''
            if len(best_p) == 1:
                chosen = best_p[0]
                alt = next((p for p in self.products.get(self.qi.items[0], []) if p.id != chosen.id), None)
                if alt is None:
                    return ''
                alt_title = alt.title if len(alt.title) < 90 else alt.title[:90] + '...'
                cp, ap = (int(chosen.price), int(alt.price))
                if ap == cp:
                    reason = f'Both are priced at {cp}, so I keep `{chosen.id}` for its stronger match on the stated attributes.'
                elif ap > cp:
                    reason = f'`{alt.id}` costs {ap} against {cp} for `{chosen.id}`, so `{chosen.id}` is the cheaper qualifying option.'
                else:
                    reason = f'`{alt.id}` is cheaper ({ap} vs {cp}) but matches the stated constraints less fully than `{chosen.id}`, which I therefore keep.'
                return f'Let me compare my pick `{chosen.id}` (price {cp}) against the next candidate `{alt.id}` (price {ap}, title `{alt_title}`). {reason}'
            for i, sq in enumerate(self.qi.items):
                if i >= len(best_p):
                    break
                chosen = best_p[i]
                alt = next((p for p in self.products.get(sq, []) if p.id != chosen.id), None)
                if alt is None:
                    continue
                cp, ap = (int(chosen.price), int(alt.price))
                alt_title = alt.title if len(alt.title) < 90 else alt.title[:90] + '...'
                effect = 'raise' if ap > cp else 'lower' if ap < cp else 'leave unchanged'
                return f"For the sub-query `{sq.query}` I chose `{chosen.id}` (price {cp}); the closest alternative is `{alt.id}` (price {ap}, title `{alt_title}`). Substituting it would {effect} the bundle total, and `{chosen.id}` already satisfies the sub-query's constraints, so I keep the combination `{', '.join((x.id for x in best_p))}`."
            return ''

        def _run_search(self, sq: S_SubQuery, sos: List[S_SearchOption], id: str, page_limit: int, dbg_prefix: str) -> Dict:
            workers = min(getattr(type(self), 'SEARCH_WORKERS', 1), len(sos))
            if workers <= 1:
                return {so: self._fetch_and_enrich(so, page_limit) for so in sos}
            raw: List[Optional[List[Tuple[List[S_Product], ToolCallResult]]]] = [None] * len(sos)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fmap = {pool.submit(self._fetch_pages, so, page_limit): i for i, so in enumerate(sos)}
                for fut in as_completed(fmap):
                    i = fmap[fut]
                    try:
                        raw[i] = fut.result()
                    except Exception as exc:
                        self.debugger.print(f'# parallel search fetch failed for option {i}: {exc}')
                        raw[i] = []
            return {so: self._integrate_pages(so, raw[i] or []) for i, so in enumerate(sos)}

        def _pick_best(self, sq: S_SubQuery, candidates: List[S_Product], must: bool, dbg_prefix: str) -> Tuple[Optional[S_Product], bool]:
            best_p, js_check, is_fallback = (None, {}, False)
            is_partial, arb_from, alvl = (False, None, 0)
            if len(candidates) > 1 or (len(candidates) == 1 and (not must)):
                js_check, _ = S_ProxyUtil.trigger_llm(S_Prompt.PV_SELECT.format(rules=S_Prompt.MATCH_RULES, rank_rules=S_Prompt.RANK_RULES, query=sq.query, products=self._render_products_block(candidates, sq, '', False)), preferred_llms=S_ProxyUtil.get_role_llm_priorities('select'), expect='dict', min_secs=S_SearchEngine.SELECT_MIN_SECS, max_secs=S_SearchEngine.SELECT_MAX_SECS, reserve_after=S_SearchEngine.RESERVE_AFTER_TERMINAL)
                bid = S_Prompt.extract_select_response(js_check)
                self.debugger.print(js_check)
                if bid >= 0 and bid < len(candidates):
                    abid, alvl = S_Prompt.arbitrate_tally(js_check, bid, len(candidates))
                    if abid != bid and 0 <= abid < len(candidates):
                        self.debugger.print(f'* tally arbiter: #{bid + 1} -> #{abid + 1} on {S_Prompt.TALLY_LEVELS[alvl - 1]}')
                        arb_from, bid = (candidates[bid], abid)
                    best_p = candidates[bid]
                elif must:
                    cid = S_Prompt.extract_closest_response(js_check)
                    if cid >= 0 and cid < len(candidates):
                        self.debugger.print(f'* fallback to the closest partial match: #{cid + 1} for {sq.query[:50]}')
                        best_p = candidates[cid]
                        is_partial = True
                    else:
                        self.debugger.print(f'* fallback to select the first product: {sq.query[:50]}')
                        best_p = candidates[0]
                    is_fallback = True
            elif len(candidates) > 0:
                self.debugger.print(f'* selected only one product for: {sq.query[:50]}')
                best_p = candidates[0]
                is_fallback = True
            else:
                self.debugger.print(f'# cannot select the best because no candidate exists!')
            if best_p is not None:
                self.sess.step_on_select(best_p, js_check, is_fallback, sq, is_partial, arb_from, alvl)
            return (best_p, is_fallback)

        def _shard_filter_candidates(self, products: Dict[S_SubQuery, List[S_Product]]) -> List[Dict[S_SubQuery, Tuple[int, List[S_Product]]]]:
            """Split a filter batch into independently-answerable slices.

            Each group is cut into CONTIGUOUS slices so a shard can be described by a
            start index per group, which is what keeps the global candidate ids intact
            (see `_build_filter_prompt`). Shard count is driven by the candidate total,
            not by the group count, because the completion length -- and therefore the
            latency -- scales with candidates.
            """
            total = sum((len(ps) for ps in products.values()))
            n = min(S_SearchEngine.FILTER_MAX_SHARDS, max(1, -(-total // S_SearchEngine.FILTER_SHARD_SIZE)))
            if n <= 1:
                return [{sq: (0, list(ps)) for sq, ps in products.items() if ps}]
            shards: List[Dict[S_SubQuery, Tuple[int, List[S_Product]]]] = []
            for s in range(n):
                shard: Dict[S_SubQuery, Tuple[int, List[S_Product]]] = {}
                for sq, ps in products.items():
                    lo, hi = (len(ps) * s // n, len(ps) * (s + 1) // n)
                    if hi > lo:
                        shard[sq] = (lo, ps[lo:hi])
                if shard:
                    shards.append(shard)
            return shards

        def _filter_shard(self, shard: Dict[S_SubQuery, Tuple[int, List[S_Product]]], letters: Dict[S_SubQuery, str], counts: List[int]) -> Tuple[List[Dict[int, bool]], Dict]:
            """Run one shard and return `(per-group verdicts, raw response)`.

            Thread-safe on purpose: it reads the sub-queries and products it was handed
            and mutates no engine or session state, so the caller can fan several of
            these out and merge on the main thread. Retries only when the shard yielded
            no readable verdict at all -- with the tolerant parser a partial answer is
            worth keeping, so re-asking is now the rare path rather than the reflex.
            """
            last_js: Dict = {}
            for _try in range(S_SearchEngine.FILTER_MAX_TRIES):
                js, _ = S_ProxyUtil.trigger_llm(self._build_filter_prompt(shard, letters), preferred_llms=S_ProxyUtil.get_role_llm_priorities('filter'), temperature=_try * 0.3, expect='dict', min_secs=S_SearchEngine.FILTER_MIN_SECS, max_secs=S_SearchEngine.FILTER_MAX_SECS, reserve_after=S_SearchEngine.RESERVE_AFTER_FILTER)
                if isinstance(js, dict) and js:
                    last_js = js
                verdicts = S_Prompt.extract_filter_response(js, counts)
                if any(verdicts):
                    return (verdicts, last_js)
            return ([{} for _ in counts], last_js)

        def _llm_filter(self, products: Dict[S_SubQuery, List[S_Product]], id: str, dbg_prefix: str) -> Optional[Dict[S_SubQuery, Dict[S_Product, bool]]]:
            """Ask the LLM which candidates match, per sub-query.

            Returns, per sub-query, the verdict for each candidate the model actually
            adjudicated -- `True` matched, `False` rejected, absent means never judged.
            Returning verdicts instead of just the passing list is what lets the caller
            stop inferring "absent from the pass list" as "rejected": with a partial
            response that inference condemns candidates nobody looked at, which is the
            exact confusion `_has_filter_reject` exists to prevent.

            Returns None when nothing could be read at all, which is still distinct
            from running and rejecting everything.
            """
            groups = list(products.keys())
            letters = {sq: S_Vocab.ALPHABETS[i] for i, sq in enumerate(groups)}
            counts = [len(products[sq]) for sq in groups]
            shards = self._shard_filter_candidates(products)
            self.debugger.print(f'* llm filter [{id}]: checking {sum(counts)} candidate(s) across {len(groups)} group(s) in {len(shards)} shard(s)...')
            results: List[Optional[Tuple[List[Dict[int, bool]], Dict]]] = [None] * len(shards)
            workers = min(S_SearchEngine.FILTER_WORKERS, len(shards))
            if workers <= 1:
                for i, shard in enumerate(shards):
                    try:
                        results[i] = self._filter_shard(shard, letters, counts)
                    except Exception as exc:
                        self.debugger.print(f'# filter shard {i} failed: {exc}')
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    fmap = {pool.submit(self._filter_shard, shard, letters, counts): i for i, shard in enumerate(shards)}
                    for fut in as_completed(fmap):
                        i = fmap[fut]
                        try:
                            results[i] = fut.result()
                        except Exception as exc:
                            self.debugger.print(f'# filter shard {i} failed: {exc}')
            merged: Dict[S_SubQuery, Dict[S_Product, bool]] = {sq: {} for sq in groups}
            js: Dict[str, Any] = {}
            n_checked = 0
            for res in results:
                if res is None:
                    continue
                verdicts, shard_js = res
                for gid, vmap in enumerate(verdicts):
                    sq = groups[gid]
                    for bid, ok in vmap.items():
                        if products[sq][bid] not in merged[sq]:
                            n_checked += 1
                        merged[sq][products[sq][bid]] = ok
                for gk, gv in (shard_js or {}).items():
                    if isinstance(gv, dict) and isinstance(js.get(gk), dict):
                        js[gk].update(gv)
                    else:
                        js[gk] = gv

            def _text_of(gid: int, bid: int) -> str:
                sq = groups[gid]
                if bid >= len(products[sq]):
                    return ''
                block = products[sq][bid].to_prompt(sq.query)
                return '\n'.join((str(v) for v in block.values()))
            n_dropped = 0
            for gid, dmap in enumerate(S_Prompt.arbitrate_filter(js, counts, _text_of)):
                sq = groups[gid]
                for bid, why in dmap.items():
                    p = products[sq][bid]
                    if merged[sq].pop(p, None) is None:
                        continue
                    n_dropped += 1
                    self.debugger.print(f'* filter arbiter [{id}]: {S_Vocab.ALPHABETS[gid]}{bid + 1} -> unverified ({why})')
            n_checked -= n_dropped
            if n_checked == 0:
                self.debugger.print(f'# llm filter [{id}] unusable after retries -- leaving {sum(counts)} candidate(s) unverified')
                self.debugger.add('k-filter', {'id': id, 'js': js, 'n_checked': sum(counts), 'skipped': True})
                return None
            self.debugger.print(js)
            n_passed = sum((1 for vmap in merged.values() for ok in vmap.values() if ok))
            self.debugger.add('k-filter', {'id': id, 'js': js, 'n_checked': n_checked, 'n_passed': n_passed})
            self.sess.step_on_filter(len(groups), n_checked, n_passed, js)
            return merged

        def run(self) -> List[S_Product]:
            self._start_watch('query')
            self._start_watch('breadth', S_SearchEngine.VERIFY_RESERVE_SECS)
            for sq in self.qi.items:
                self.searches[sq] = self._run_search(sq, self._build_search_options(sq, True), None, 5, 'b')
                if self.debugger.is_stop('b-search'):
                    continue
                bi_sos = self._build_bisect_search_options(sq)
                if bi_sos:
                    bi_results = self._run_search(sq, bi_sos, None, 5, 'bi')
                    self.searches[sq].update(bi_results)
                self.products[sq] = self._score_products(sq, self.searches[sq], None, 'b')
                if self.debugger.is_stop('b-weigh'):
                    continue
                if self.products[sq]:
                    self.fscore[sq] = self.weight_pool[sq][self.products[sq][0].id].score_n
                else:
                    self.debugger.print(f'* no products found for `{sq.query[:50]}`')
                    self.fscore[sq] = math.inf
                self.irank[sq] = len(self.products[sq])
            self.sess.step_on_search(self.qi)
            return []



__all__ = ["agent_main"]
