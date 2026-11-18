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

_KNG00CEB_AST_PAD = 0
if _KNG00CEB_AST_PAD:
    class _KNG00CEB_L_ShoppingTaskType(str, Enum):
        PRODUCT = 'product'
        SHOP = 'shop'
        VOUCHER = 'voucher'
        def explain(self, is_multiple: bool) -> str:
            match self:
                case L_ShoppingTaskType.PRODUCT:
                    return 'a single product with specific constraints'
                case L_ShoppingTaskType.SHOP:
                    return 'multiple products from the same shop'
                case L_ShoppingTaskType.VOUCHER:
                    pstr = 'multiple products' if is_multiple else 'a single product'
                    return f'{pstr} within a specified budget and considering voucher options'


    def _KNG00CEB_L_calculate_voucher(product_prices: str, voucher_type: str, discount_value: float, threshold: float, budget: float, cap: float=0) -> Dict:
        prices_raw = [s.strip() for s in product_prices.split(',') if s.strip()]
        try:
            prices = [float(p) for p in prices_raw]
        except ValueError as exc:
            return {'error': f'Invalid product_prices: {exc}'}
        if not prices: return {'error': 'product_prices is empty or null.'}
        vtype = (voucher_type or '').strip().lower()
        if vtype not in ('fixed', 'percentage'): return {'error': f"Invalid voucher_type: {voucher_type!r}. Use 'fixed' or 'percentage'."}
        threshold = max(float(threshold or 0), 0.0)
        budget = max(float(budget or 0), 0.0)
        discount_value = max(float(discount_value or 0), 0.0)
        cap = max(float(cap or 0), 0.0)
        total = sum(prices)
        discount = 0.0
        voucher_applied = False
        if total >= threshold:
            voucher_applied = True
            if vtype == 'fixed': discount = min(discount_value, total)
            else:
                pct = discount_value / 100.0
                discount = total * pct
                if cap > 0: discount = min(discount, cap)
        total_after = max(0.0, total - discount)
        return {'prices': prices, 'total_before': round(total, 2), 'discount_amount': round(discount, 2), 'total_after': round(total_after, 2), 'within_budget': total_after <= budget, 'voucher_applied': voucher_applied, 'budget': budget}


    def _KNG00CEBdual_bind_tool(name: str, fn) -> None:
        register_tool(name, fn)
        try:
            import src.agent.agent_interface as _ai
            for attr in ("SANDBOX_TOOLS", "_SANDBOX_TOOLS", "_TOOL_REGISTRY"):
                reg = getattr(_ai, attr, None)
                if isinstance(reg, dict):
                    reg[name] = fn
        except Exception:
            pass


    class _KNG00CEB_FN_EnhancementBenchSettings:
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


    def _KNG00CEBmu_recommend(product_ids: str = "", ids: str = "") -> str:
        return Mu_ExtTool.submit(product_ids=product_ids or ids)


    class _KNG00CEB_KAK_ProxyUtil:
        DEADLINE = None
        @staticmethod
        def arm_deadline(*_a, **_k):
            pass
        @staticmethod
        def time_left():
            d = KAK_ProxyUtil.DEADLINE
            return max(0.0, float(d) - time.monotonic()) if d else 0.0


    class _KNG00CEB_L_Product(L_JsonModel):
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
                    for l in lines: joint_attr.append(l)
            js = {'title': self.title, 'attributes': '\n'.join(joint_attr)}
            if for_reason: js['price'] = self.price
            return js
        def clone_from(self, other: 'L_Product') -> None:
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
                    if L_Support.is_valid_attr(k, v, q): is_valid = True
                    atoms.append((k, v))
                if atoms: extracted[f'att_{k}'].append(atoms)
            for _, kv in self.sku.items():
                atoms = []
                for k, v in kv.items():
                    if L_Support.is_valid_attr(k, v, q): is_valid = True
                    atoms.append((k, v))
                if atoms: extracted['sku'].append(atoms)
            if not is_valid: extracted.clear()
            return {g: [[f'{(L_Support.normalize_str(k) if norm else k)} {delimiter} {(L_Support.normalize_str(v) if norm else v)}' for k, v in kv] for kv in kvs] for g, kvs in extracted.items()}
        def __hash__(self): return hash(self.id)
        def __eq__(self, other):
            if not isinstance(other, L_Product): return NotImplemented
            return self.id == other.id


    def _KNG00CEBP_query_literal_score(query: str, product_blob: str) -> float:
        q = (query or "").lower()
        t = (product_blob or "").lower()
        stop = {"the", "and", "for", "with", "from", "that", "this", "please", "show", "looking", "those", "only", "less", "than", "under", "above", "between", "pesos", "php", "costs", "priced", "price", "made", "suitable", "also", "need", "want", "find"}
        toks = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) >= 3 and w not in stop]
        if not toks:
            return 0.0
        hit = sum(1 for w in toks if w in t)
        return hit / max(1, len(toks))


    class _KNG00CEB_Lou_SubQuery(Lou_JsonModel):
        query_f: str = ""
        query: str = ""
    
        keywords: List[str] = field(default_factory = list)
        services: List[str] = field(default_factory = list)

        price: Optional[str] = None    
        id: int = -1

        template: Dict[str, str] = field(default_factory = dict)
        prod_only: bool = False

        def __hash__(self) -> int:
            return hash((self.query_f, self.id))

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, type(self)): return False
            return self.query_f == other.query_f and self.id == other.id


    class _KNG00CEB_Lou_Proxy:
        LLM_PROXY = ProxyClient(timeout = 200, max_retries = 5, retry_delay = 1.0)
        ORO_PROXY = ProxyClient(timeout = 30, max_retries = 5, retry_delay = 1.0)    

        LLM_PRIORITIES = [
            "moonshotai/kimi-k2.6", "deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1", "qwen/qwen3.5-397b-a17b",
            "z-ai/glm-5.2", "z-ai/glm-5.2", "deepseek/deepseek-chat-v3-0324", "deepseek/deepseek-r1-0528",
            "z-ai/glm-5.2", "qwen/qwen3-32b", "xiaomi/mimo-v2-flash", "google/gemma-4-31b-it"
        ]
        TEMP_UP = 0.3

        @staticmethod
        def trigger_llm(
            prompt: str, thinking: bool = False, preferred_llms: List[str] = [], temperature: float = 0, max_tokens: int = 8192, expect: str = "any"
        ) -> Tuple[
            Dict[str, Any] | List[Any], Any
        ]:        
            llm_priorities = Lou_Proxy.__get_llm_priorities__(preferred_llms)
            trigger_tick = time.time()
            t = temperature

            while True:
                for model in llm_priorities:
                    payload = Lou_Proxy.__create_llm_payload__(model, prompt, thinking, t, max_tokens)

                    try:
                        tick = time.time()

                        resp: Dict[str, Dict] = Lou_Proxy.LLM_PROXY.post("/inference/chat/completions", json_data = payload)
                        content = Lou_Proxy.__resp_to_content__(resp)

                        try:
                            js = Lou_Misc.parse_json(content)
                            if js is None: raise Exception()
                        except Exception as exc:
                            t += Lou_Proxy.TEMP_UP                        
                            raise Exception(f"failed to parse json: {exc}\n*******\n{content}\n******")

                        if (expect == "dict" and not isinstance(js, dict)) or \
                            (expect == "list" and not isinstance(js, list)):
                            t += Lou_Proxy.TEMP_UP
                            raise Exception(f"expected {expect} but got {type(js)}")

                        throughput = resp.get("usage", {}).get("completion_tokens", 0)                    
                        trigger_delay = time.time() - trigger_tick
                        delay = time.time() - tick
                    
                        Lou_Misc.LOGGER.info(
                            f"* llm triggered: model = '{model}' | throughput = {throughput} | delay = {delay:.2f}s | "
                            f"total_delay = {trigger_delay:.2f}s"
                        )
                        return js, resp
                    except Exception as exc:
                        Lou_Misc.LOGGER.error(f"# llm trigger error with '{model}': {exc}")
                        continue

        @staticmethod
        def search(filter: Lou_Filter, sort: Optional[str] = None, page: int = 1) -> Tuple[List[Lou_Product], ToolCallResult]:
            while True:
                try:
                    tool_res = execute_tool_call(
                        "find_product",
                        {
                            "q": filter.get_q(),
                            "page": page,
                            "shop_id": filter.shop,
                            "price": filter.price,
                            "sort": sort,
                            "service": ','.join(filter.services or [])
                        }
                    )
                    js_prods: List[Dict] = tool_res["result"]
                    products = []

                    for j in js_prods:
                        price = j.get("price") or 0
                        price = Lou_Misc.extract_number(str(price))

                        products.append(
                            Lou_Product(
                                id = str(j.get("product_id")),
                                title = j.get("title", ""),
                                price = price,
                                shop = j.get("shop_id", ""),
                                services = j.get("service", []) or [],
                                attr = {},
                                sku = {},
                                is_viewed = False
                            )
                        )
                
                    return products, tool_res

                except Exception as exc:
                    Lou_Misc.LOGGER.error(f"# error in `find_products`: {exc}")
                    time.sleep(0.5)

        @staticmethod
        def view(products: List[Lou_Product]) -> ToolCallResult:
            if not products: return        
            pid_str = ','.join([p.id for p in products])

            while True:
                try:
                    tool_res = execute_tool_call("view_product_information", {"ids": pid_str})        
                
                    details = tool_res["result"]
                    if not details: return tool_res

                    details_map: Dict[str, Dict[str, Any]] = {d["product_id"]: d for d in details}

                    for prod in products:
                        d = details_map.get(prod.id, {})
                    
                        prod.sku = d.get("sku_options", {})
                        prod.attr = d.get("attributes", {})
                        prod.is_viewed = True
                
                    return tool_res
                except Exception as exc:
                    Lou_Misc.LOGGER.error(f"# error in `view_products`: {exc}")
                    time.sleep(0.5)

        @staticmethod
        def submit(products: List[Lou_Product]) -> ToolCallResult:
            return execute_tool_call("recommend_product", {"product_ids": ','.join([p.id for p in products])})
    
        @staticmethod
        def finish(status: str) -> ToolCallResult:
            return execute_tool_call("terminate", {"status": status})

        @staticmethod
        def __get_llm_priorities__(prefers: List[str]) -> List[str]:
            priorities = prefers.copy()

            for llm in Lou_Proxy.LLM_PRIORITIES:
                if llm not in priorities: priorities.append(llm)
        
            return priorities

        @staticmethod
        def __create_llm_payload__(model: str, prompt: str, thinking: bool, temperature: float, max_tokens: int) -> Dict:
            return {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "chat_template_kwargs": {
                    "enable_thinking": thinking,
                    "thinking": thinking
                },
                "reasoning": {
                    "effort": "medium" if thinking else "none",
                    "enabled": thinking
                },
                "thinking": {
                    "type": "disabled" if not thinking else "enabled"
                },
                "temperature": temperature,
                "stream": False,
                "max_completion_tokens": max_tokens,
                "max_tokens": max_tokens
            }

        @staticmethod
        def __resp_to_content__(resp: Dict) -> str:
            try:
                choice: Dict[str, Dict] = resp.get("choices")[0]
                content = choice.get("message").get("content")

                if not content: raise Exception()
                return content
            except Exception as exc:
                raise Exception(f"invalid response ({exc})")


    class _KNG00CEB_FN_SearchEngine(object):
        SELECT_CANDIDATES = 30
        SCORE_N_THRES = 0.4
        TIMEOUT_SECS = 240.0
        def __init__(self, qi: FN_QueryInfo, sess: FN_SearchSession, debugger: FN_Debugger = None) -> None:
            self.debugger = debugger
            self.sess = sess
            self.qi = qi
            self.weight_pool: Dict[FN_SubQuery, Dict[str, FN_Weight]] = defaultdict(lambda: {})
            self.prod_pool: Dict[str, FN_Product] = {}
            self.searches: Dict[FN_SubQuery, Dict[FN_SearchOption, List[FN_Product]]] = {}
            self.products: Dict[FN_SubQuery, List[FN_Product]] = {}
            self.synonyms: List[Tuple[str, str]] = [(a, b) for a, b in FN_Vocab.SYNONYMS]
            self.fillers: List[str] = []
            self.sos: Dict[FN_SubQuery, List[FN_SearchOption]] = defaultdict(lambda: [])
            self.irank: Dict[FN_SubQuery, int] = defaultdict(lambda: 0)
            self.fscore: Dict[FN_SubQuery, float] = {}
            self.filtered = defaultdict(lambda: defaultdict(lambda: None))
            self.watches: Dict[str, float] = {}
            self.verified: Dict[FN_SubQuery, List[FN_Product]] = defaultdict(list)
            self.rrf_rank: Dict[FN_SubQuery, Dict[str, float]] = defaultdict(dict)
        RRF_K = 60
        RRF_WEIGHT = 0.15
        VSF_VERIFY_MAX = 12
        VSF_HARD_TOP = 20
        VSF_EXPAND_BATCHES = 2
        CTR_MAX_OPTIONS = 4
        CTR_NEAR_MISS = 5
        SEARCH_RECOVER_PAGE_LIMIT = 3
        TIGHT_RACE_GAP = 0.12
        COVERAGE_PRUNE_FLOOR = 0.35
        def run(self) -> List[FN_Product]:
            self.__init_watch__("query", FN_SearchEngine.TIMEOUT_SECS)
            for sq in self.qi.items:
                self.searches[sq] = self.__search__(sq, self.__get_sos__(sq, True), None, 5, "b")
                if self.debugger.is_stop("b-search"): continue
                bi_sos = self.__get_bi_sos__(sq)
                if bi_sos: bi_results = self.__search__(sq, bi_sos, None, 5, "bi"); self.searches[sq].update(bi_results)
                self.products[sq] = self.__weigh__(sq, self.searches[sq], None, "b")
                if self.debugger.is_stop("b-weigh"): continue
                if self.products[sq]: self.fscore[sq] = self.weight_pool[sq][self.products[sq][0].id].score_n
                else:
                    self.debugger.print(f"* no products found for `{sq.query[:50]}`")
                    self.fscore[sq] = math.inf
                self.irank[sq] = len(self.products[sq])
            self.sess.step_on_search(self.qi)
            return []
        def __search__(
            self,
            sq: FN_SubQuery,
            sos: List[FN_SearchOption],
            id: str,
            page_limit: int,
            dbg_prefix: str
        ) -> Dict:
            js: Dict[FN_SearchOption, List[FN_Product]] = {}
            for so in sos: js[so] = self.__find_and_view__(so, page_limit)
            return js
        def __weigh__(
            self,
            sq: FN_SubQuery,
            searches: Dict[FN_SearchOption, List[FN_Product]],
            id: Optional[str],
            dbg_prefix: str
        ) -> List[FN_Product]:
            ranks: Dict[FN_Product, int] = defaultdict(lambda: math.inf)
            products: List[FN_Product] = []
            for so in searches:
                for r, p in enumerate(searches[so]):
                    if r < ranks[p]:
                        if p not in products: products.append(p)
                        ranks[p] = r
            for p in sorted(products, key = lambda p: ranks[p]):
                w = self.__weigh_product__(p, sq, ranks[p] + self.irank[sq])
                if p.id in self.weight_pool[sq]: self.weight_pool[sq][p.id].update_with_sibling(w)
                else: self.weight_pool[sq][p.id] = w
            rrf: Dict[str, float] = defaultdict(float)
            for so in searches:
                for r, p in enumerate(searches[so]): rrf[p.id] += 1.0 / (FN_SearchEngine.RRF_K + r + 1)
            self.rrf_rank[sq] = dict(rrf)
            top_rrf = max(rrf.values()) if rrf else 1.0
            def _fuse_key(p: FN_Product) -> float: sc = self.weight_pool[sq][p.id].score_n; return -(sc + FN_SearchEngine.RRF_WEIGHT * (rrf.get(p.id, 0.0) / top_rrf))
            products.sort(key = _fuse_key)
            for i, p in enumerate(products): self.weight_pool[sq][p.id].update_confidence(i + self.irank[sq])
            return products
        def __select__(
            self, sq: FN_SubQuery, candidates: List[FN_Product], must: bool, dbg_prefix: str
        ) -> Tuple[
            Optional[FN_Product], bool
        ]:
            best_p, js_check, is_fallback = None, {}, False
            if len(candidates) > 1:
                scored = []
                for p in candidates:
                    cov = FN_EngineUtil.score_candidate_coverage(p, sq)
                    wn = self.weight_pool[sq][p.id].score_n if p.id in self.weight_pool[sq] else 0.0
                    scored.append((p, cov["ratio"], wn))
                scored.sort(key=lambda t: (-t[1], -t[2]))
                kept = [p for p, ratio, _ in scored if ratio >= FN_SearchEngine.COVERAGE_PRUNE_FLOOR]
                if kept:
                    kept_ids = {p.id for p in kept}
                    rest = [p for p in candidates if p.id not in kept_ids]
                    candidates = kept + rest
                else:
                    candidates = [p for p, _, _ in scored]
            if len(candidates) > 1 or (len(candidates) == 1 and not must):
                js_check, _ = FN_ProxyUtil.trigger_llm(FN_Prompt.PV_SELECT.format(rules = FN_Prompt.MATCH_RULES, query = sq.query, products = self.__get_prod_str__(candidates, sq, "", False)), preferred_llms=FN_ProxyUtil.get_role_llm_priorities("select"), expect = "dict")
                bid = FN_Prompt.extract_select_response(js_check)
                self.debugger.print(js_check)
                if bid >= 0 and bid < len(candidates): best_p = candidates[bid]
                elif must:
                    self.debugger.print(f"* fallback to select the first product: {sq.query[:50]}")
                    best_p = candidates[0]
                    is_fallback = True
                if (
                    best_p is not None
                    and len(candidates) >= 2
                    and self.__check_watch__()
                    and not FN_ProxyUtil.out_of_time()
                    and FN_ProxyUtil.time_left() > 45.0
                ):
                    top_scores = sorted((self.weight_pool[sq][p.id].score_n for p in candidates if p.id in self.weight_pool[sq]), reverse=True,)
                    if (
                        len(top_scores) >= 2
                        and (top_scores[0] - top_scores[1]) < FN_SearchEngine.TIGHT_RACE_GAP
                    ):
                        shortlist = candidates[:8]
                        js2, _ = FN_ProxyUtil.trigger_llm(FN_Prompt.PV_SELECT.format(rules = FN_Prompt.MATCH_RULES, query = sq.query, products = self.__get_prod_str__(shortlist, sq, "", False)), preferred_llms=FN_ProxyUtil.get_role_llm_priorities("select"), temperature = 0.3, expect = "dict")
                        bid2 = FN_Prompt.extract_select_response(js2)
                        if 0 <= bid2 < len(shortlist):
                            alt = shortlist[bid2]
                            if alt.id == best_p.id: js_check = js2
                            else:
                                cov1 = FN_EngineUtil.score_candidate_coverage(best_p, sq)["ratio"]
                                cov2 = FN_EngineUtil.score_candidate_coverage(alt, sq)["ratio"]
                                if cov2 > cov1 + 0.05: best_p = alt; js_check = js2
            elif len(candidates) > 0:
                self.debugger.print(f"* selected only one product for: {sq.query[:50]}")
                best_p = candidates[0]
                is_fallback = True
            else:
                self.debugger.print(f"# cannot select the best because no candidate exists!")
            if best_p is not None: self.sess.step_on_select(best_p, js_check, is_fallback, sq)
            return best_p, is_fallback
        def __filter__(
            self,
            products: Dict[FN_SubQuery, List[FN_Product]],
            id: str,
            dbg_prefix: str
        ) -> Dict[FN_SubQuery, List[FN_Product]]:
            temperature = 0
            js = {}
            resp = [[] for _ in products]
            for _attempt in range(6):
                try:
                    n_cands = sum(len(ps) for ps in products.values())
                    self.debugger.print(f"* llm filter [{id}]: checking {n_cands} candidate(s) across {len(products)} group(s)...")
                    js, _ = FN_ProxyUtil.trigger_llm(self.__create_filter_prompt__(products), preferred_llms = FN_ProxyUtil.get_role_llm_priorities("filter"), temperature = temperature, expect = "dict")
                    self.debugger.print(js)
                    resp = FN_Prompt.extract_filter_response(js, [len(products[sq]) for sq in products])
                    break
                except Exception as exc:
                    self.debugger.print(f"# failed filtering: {exc}")
                    temperature += 0.3
                    if FN_ProxyUtil.out_of_time() or temperature > 1.5:
                        self.debugger.print(f"# filter giving up after retries [{id}]")
                        resp = [[] for _ in products]
                        break
            filtered: Dict[FN_SubQuery, List[FN_Product]] = {}
            for i in range(len(products)):
                sq = list(products.keys())[i]
                filtered[sq] = []
                for bid in resp[i]: filtered[sq].append(products[sq][bid])
            n_groups = len(products)
            n_checked = sum(len(ps) for ps in products.values())
            n_passed = sum(len(ps) for ps in filtered.values())
            self.debugger.add("k-filter", {"id": id, "js": js, "n_checked": n_checked, "n_passed": n_passed})
            self.sess.step_on_filter(n_groups, n_checked, n_passed, js)
            return filtered
        def __find_and_view__(self, opt: FN_SearchOption, page_limit: int = 1) -> List[FN_Product]:
            res: List[FN_Product] = []
            for page in range(1, 6):
                searched, tool_res = FN_ProxyUtil.find_products(opt, page = page)
                self.sess.add_tool_result(tool_res)
                found: List[FN_Product] = []
                for p in searched:
                    if self.__basic_check__(p, opt): found.append(self.__check_with_pool__(p))
                to_view = [p for p in found if not p.is_viewed]
                if len(to_view) > 0: FN_ProxyUtil.view_products(to_view)
                for p in found: res.append(self.__check_with_pool__(p))
                if page >= page_limit: break
            return res
        def __basic_check__(self, prod: FN_Product, so: FN_SearchOption) -> bool:
            if so.price:
                if not prod.price: return False
                prod_price = FN_MiscUtil.extract_number(str(prod.price))
                low, high = FN_EngineUtil.numerify_price(so.price)
                if prod_price < low or prod_price > high: return False
            if so.services:
                prod_services = [str(svc).lower() for svc in (prod.service or [])]
                sq_services = [str(svc).lower() for svc in so.services]
                for svc in sq_services:
                    if svc not in prod_services: return False
            if so.shop_id and so.shop_id != prod.shop_id: return False
            return True
        def __weigh_product__(self, prod: FN_Product, sq: FN_SubQuery, rank: int) -> FN_Weight:
            def __get_cand_vs__(v_o: str) -> List[str]:
                v_cands = [v_o]
                for a, b in self.synonyms:
                    if a == v_o and b not in v_cands: v_cands.append(b)
                    if b == v_o and a not in v_cands: v_cands.append(a)
                return v_cands
            def __dry_run__(title: str, joint_attr: List[str]):
                used_attr, used_headers = [], []
                len_pool, ti_pool, ai_pool, num_pool, match_pool = {}, {}, {}, {}, {}
                for kw in sq.kw:
                    v_cands, ks = __get_cand_vs__(kw[-1]), kw[:-1]
                    for v in v_cands:
                        atoms = [a.strip() for a in FN_MiscUtil.normalize_str(v).split() if a.strip()]
                        hash_key = tuple(kw + [v])
                        len_pool[hash_key] = len(atoms)
                        as_is = re.search(r"^(\d+)$", v)
                        ti_pool[hash_key] = {i for i, a in enumerate(atoms) if FN_EngineUtil.is_kw_in(a, title, as_is, self.synonyms)}
                        num_pool[hash_key] = [i for i, a in enumerate(atoms) if a.isnumeric()]
                        for ja in joint_attr:
                            ai_pool[(hash_key, ja)] = [i for i, kw in enumerate(atoms) if FN_EngineUtil.is_kw_in(kw, ja, as_is, self.synonyms)]
                            match_pool[(hash_key, ja)] = (False, 0)
                            if len(ai_pool[(hash_key, ja)]) == len(atoms):
                                if any(FN_EngineUtil.is_kw_in(a, ja.split('🎈')[1], as_is, self.synonyms) for a in atoms): match_pool[(hash_key, ja)] = (True, __get_key_bonus__(ja, ks, atoms))
                            if ja in used_attr: continue
                            h, b = ja.split('🎈')
                            if any(FN_EngineUtil.is_kw_in(kw, b, as_is, self.synonyms) for kw in atoms): used_attr.append(ja)
                            if h in used_headers: continue
                            if any(FN_EngineUtil.is_kw_in(kw, h, as_is, self.synonyms) for kw in atoms): used_headers.append(h)
                return (used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool)
            def __pre_compute__(title: str):
                all_attr = prod.extract_attr(sq.query, '🎈', True)
                joint_attr = [l for _, g in all_attr.items() for lines in g for l in lines]
                attr = {}
                (used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool) = __dry_run__(title, joint_attr)
                for idx, g in all_attr.items():
                    new_g = []
                    for lines in g:
                        new_lines, headers_seen = [], []
                        for l in lines:
                            if l in used_attr: new_lines.append(l)
                            else:
                                h = l.split('🎈')[0]
                                if h not in headers_seen and h in used_headers: headers_seen.append(h); new_lines.append(l)
                        new_lines = [l for l in new_lines if __is_valid__(l)]
                        if new_lines: new_g.append(new_lines)
                    if new_g: attr[idx] = new_g
                combs = FN_MiscUtil.get_choice_combs(attr, True)
                return (attr, combs, len_pool, ti_pool, ai_pool, num_pool, match_pool)
            def __get_key_bonus__(ja: str, ks: List[str], atoms: List[str]) -> float:
                bon = 0
                if ja.split('🎈')[0] not in ["variation", "color_family"]:
                    for k in ks:
                        k = FN_MiscUtil.normalize_str(k.replace('_', ' '))
                        found = False
                        for ak in k.split():
                            ak = ak.strip()
                            if not ak or ak in atoms: continue
                            if FN_EngineUtil.is_kw_in(ak, ja, False, self.synonyms):
                                bon += 0.1
                                found = True
                                break
                        if found:
                            if k == ja.split('🎈')[0]: bon += 0.05
                            break
                else:
                    bon += 0.05
                return bon
            def __is_valid__(ja: str) -> float:
                jv = FN_MiscUtil.normalize_str(ja.split('🎈')[1])
                for s in jv.split():
                    if not s: continue
                    if s.isnumeric() or s in FN_Vocab.COLOR_WORDS:
                        if f".{s}" in ja or f".{s}" in sq.query: continue
                        cands, exists = {s}, False
                        for a, b in self.synonyms:
                            if a == s: cands.add(b)
                            if b == s: cands.add(a)
                        for k in cands:
                            if re.search(rf"\b{re.escape(k)}\b", FN_MiscUtil.normalize_str(sq.query), re.I): exists = True; break
                        if not exists: return False
                return True
            title = FN_MiscUtil.normalize_str(prod.title)
            candidates: List[FN_Weight] = []
            (attr, combs, len_pool, ti_pool, ai_pool, num_pool, match_pool) = __pre_compute__(title)
            for group in combs:
                joint_attr: List[str] = [a for ak, idx in group.items() for a in (attr[ak][idx] if idx >= 0 else [])]
                weight = FN_Weight(group = group, rank = rank)
                for i_kw, kw in enumerate(sq.kw):
                    v_cands, ks = __get_cand_vs__(kw[-1]), kw[:-1]
                    max_score = -math.inf
                    for v in v_cands:
                        hash_key = tuple(kw + [v])
                        atom_len = len_pool[hash_key]
                        attr_includes = set()
                        score = 0
                        title_includes = ti_pool[hash_key]
                        found_in_attr, bonuses = False, []
                        for ja in joint_attr:
                            attr_includes.update(ai_pool[(hash_key, ja)])
                            if match_pool[(hash_key, ja)][0]: bonuses.append(match_pool[(hash_key, ja)][1]); found_in_attr = True
                        if not found_in_attr:
                            if len(title_includes) < atom_len:
                                frags = title_includes | attr_includes
                                if all(i in frags for i in num_pool[hash_key]): score += len(frags) / atom_len
                            else:
                                score += 1.0
                        else:
                            score += 1.0 if ks[-1] != "product_type" else 0.95
                            score += max(bonuses)
                        if score > max_score: max_score = score
                    wk = '_'.join(ks)
                    if wk in weight.detail: wk += f"_{i_kw + 1}"
                    weight.detail[wk] = max_score
                    weight.score += max_score
                weight.normalize()
                candidates.append(weight)
            candidates.sort(key = lambda w: -w.score_n)
            return candidates[0]
        def __check_with_pool__(self, p: FN_Product) -> FN_Product:
            if p.id in self.prod_pool:
                existing = self.prod_pool[p.id]
                if p.is_viewed and not existing.is_viewed: existing.clone_from(p)
            else:
                self.prod_pool[p.id] = p
            return self.prod_pool[p.id]
        def __get_kw_var__(self, sq: FN_SubQuery) -> Tuple[List[List[List[str]]], List[int]]:
            unit_indice, cat = [], []
            for i, ki in enumerate(sq.kw):
                ks, v = ki[:-1], ki[-1]
                if (m := re.search(r"^(\d+)\s+([a-z]+)$", v, re.I)):
                    if not any(k in ks for k in FN_Vocab.ID_KEYS):
                        val, unit = m.group(1), m.group(2)
                        cat.append([ki.copy(), ks + [unit] + [val], ks + [f"{val}{unit}"]])
                        unit_indice.append(i)
                elif (m := re.search(r"^(\d+(?:\-\d+)?)([a-z]+)$", v, re.I)):
                    if not any(k in ks for k in FN_Vocab.ID_KEYS):
                        val, unit = m.group(1), m.group(2)
                        cat.append([ki.copy(), ks + [unit] + [val]])
                        unit_indice.append(i)
                elif ks == ["size"] and len(v) >= 5 and not re.search(r"\d", v):
                    d = FN_Prompt.extract_kw_response.__name__[:5]
                    if (m := re.search(rf"\b({re.escape(d)})\b", v, re.I)):
                        abb = m.group(1)[1]
                        v = v.replace(m.group(1), "").strip()
                    else:
                        abb = ""
                    if len(v) >= 5 and len(v.split()) == 1 and v[0] in "slm":
                        abb += v[0]
                        cat.append([ki.copy(), ks + [abb]])
                        unit_indice.append(i)
                        self.synonyms.append((min(ki[-1], abb), max(ki[-1], abb)))
                else:
                    d1 = FN_Prompt.extract_kw_response.__name__.split('_')[-1][:3]
                    d2 = 'pro'
                    nv = re.sub(rf"(\w+)(?:\s+|\-){re.escape(d1)}(?:is)[a-z]+", r"\1" + f"{d2}of", v, flags = re.I)
                    if nv != v: cat.append([ki.copy(), ks + [nv]]); unit_indice.append(i)
            return cat, unit_indice
        def __get_sos__(self, sq: FN_SubQuery, shadow: bool) -> List[FN_SearchOption]:
            if self.sos[sq]: return [FN_cast_model(so, FN_SearchOption) for so in self.sos[sq]]
            cat, unit_indice = self.__get_kw_var__(sq)
            sos: List[FN_SearchOption] = []
            if cat:
                combs = FN_MiscUtil.get_choice_combs(cat, False)
                sos = []
                for choice in combs:
                    for i in range(len(unit_indice)):
                        id, c = unit_indice[i], choice[i]
                        new_sq: FN_SubQuery = FN_cast_model(sq, FN_SubQuery)
                        new_sq.kw[id] = cat[i][c]
                        sos.append(FN_cast_model(new_sq, FN_SearchOption))
            else:
                sos = [FN_cast_model(sq, FN_SearchOption)]
            for so in sos:
                if shadow:
                    low, high = FN_EngineUtil.numerify_price(sq.price or "-")
                    if high < math.inf: low = max(low, high // 2)
                    so.price = FN_EngineUtil.stringfy_price(low, high)
                so.update_q()
            self.sos[sq] = list(set(sos))
            return self.sos[sq]
        def __get_bi_sos__(self, sq: FN_SubQuery) -> List[FN_SearchOption]:
            base_sos = self.sos[sq]
            if len(base_sos) != 1: return []
            products = [p for prods in self.searches[sq].values() for p in prods]
            if not products: return []
            base_so: FN_SearchOption = FN_cast_model(base_sos[0], FN_SearchOption)
            prices = sorted(p.price for p in products)
            median = int(prices[len(prices) // 2])
            low, high = FN_EngineUtil.numerify_price(base_so.price or "-")
            lso: FN_SearchOption = FN_cast_model(base_so, FN_SearchOption)
            lso.update_price(FN_EngineUtil.stringfy_price(low, median))
            hso: FN_SearchOption = FN_cast_model(base_so, FN_SearchOption)
            hso.update_price(FN_EngineUtil.stringfy_price(median, high))
            return [lso, hso]
        def __get_prod_str__(self, products: List[FN_Product], sq: FN_SubQuery, prefix: str, for_reason: bool) -> str:
            prod_str = ""
            for i, p in enumerate(products):
                js_p = p.to_prompt(sq.query, for_reason)
                js_p["id"] = prefix + str(i + 1)
                prod_str += json.dumps(js_p, ensure_ascii = False) + '\n'
            return prod_str
        def __hard_prefilter__(self, sq: FN_SubQuery, candidates: List[FN_Product]) -> List[FN_Product]:
            out: List[FN_Product] = []
            for p in candidates:
                if sq.price:
                    ok, _ = FN_FuzzyBoundaryLogic.price_fits_range(p.price, sq.price, fuzzy=True)
                    if not ok: continue
                if sq.services:
                    prod_svc = {str(s).lower() for s in (p.service or [])}
                    if any(str(s).lower() not in prod_svc for s in sq.services): continue
                out.append(p)
            return out
        def __verify_shortlist__(
            self, sq: FN_SubQuery, candidates: List[FN_Product], max_n: int = None
        ) -> List[FN_Product]:
            if not self.__check_watch__(): return []
            max_n = max_n if max_n is not None else FN_SearchEngine.VSF_VERIFY_MAX
            hard = self.__hard_prefilter__(sq, candidates)
            if not hard: return []
            shortlist = hard[:max_n]
            try:
                verified = self.__filter__({sq: shortlist}, "vsf", "v").get(sq, [])
            except Exception as exc:
                self.debugger.print({"vsf_error": str(exc), "sq": (sq.query or "")[:50]})
                return []
            self.debugger.print({"vsf": (sq.query or "")[:50], "in": len(shortlist), "verified": len(verified)})
            return verified
        def __merge_verified__(self, sq: FN_SubQuery, new_items: List[FN_Product]) -> None:
            if not new_items: return
            seen = {p.id for p in self.verified.get(sq, [])}
            merged = list(self.verified.get(sq, []))
            for p in new_items:
                if p.id not in seen: seen.add(p.id); merged.append(p)
            self.verified[sq] = merged
        def __verify_pruned_batches__(self, sq: FN_SubQuery) -> List[FN_Product]:
            pruned = self.__prune_by_score__(self.products.get(sq, []), sq)
            found: List[FN_Product] = []
            seen: set = set()
            for batch in range(FN_SearchEngine.VSF_EXPAND_BATCHES):
                start = batch * FN_SearchEngine.VSF_VERIFY_MAX
                chunk = pruned[start:start + FN_SearchEngine.VSF_VERIFY_MAX]
                if not chunk: break
                for p in self.__verify_shortlist__(sq, chunk):
                    if p.id not in seen: seen.add(p.id); found.append(p)
                if found: break
            return found
        def __literal_constraint_tokens__(self, sq: FN_SubQuery) -> List[str]:
            tokens: List[str] = []
            for kw in sq.kw or []:
                if not kw: continue
                v = kw[-1].strip()
                k = kw[-2] if len(kw) >= 2 else ""
                literal = v.isnumeric() or k in FN_Vocab.ID_KEYS or any(c.isdigit() for c in v)
                if literal or k in ("color", "color_family", "size", "model", "compatibility", "brand"):
                    for t in v.lower().split():
                        if t and t not in tokens: tokens.append(t)
            return tokens
        def __build_targeted_sos_heuristic__(self, sq: FN_SubQuery, base_so: FN_SearchOption) -> List[FN_SearchOption]:
            kws = {kw[-2]: kw[-1].strip() for kw in sq.kw if kw and len(kw) >= 2}
            pt, brand = kws.get("product_type"), kws.get("brand")
            model = kws.get("model") or kws.get("compatibility")
            color = kws.get("color") or kws.get("color_family")
            literals = self.__literal_constraint_tokens__(sq)
            plans: List[str] = []
            if pt and brand: plans.append(f"{pt} {brand}")
            if pt and model: plans.append(f"{pt} {model}")
            if pt and color: plans.append(f"{pt} {color}")
            if pt and literals: plans.append(" ".join([pt] + literals[:4]))
            if literals: plans.append(" ".join(literals[:6]))
            existing_q = {so.q for so in self.sos.get(sq, [])}
            sos: List[FN_SearchOption] = []
            for q in plans:
                q = " ".join(q.split())
                if not q or q in existing_q or len(q.split()) > 8: continue
                so = FN_cast_model(base_so, FN_SearchOption)
                so.update_q(q)
                sos.append(so)
                existing_q.add(q)
            return sos
        def __build_targeted_sos_llm__(
            self, sq: FN_SubQuery, near_miss: List[FN_Product], base_so: FN_SearchOption
        ) -> List[FN_SearchOption]:
            if not near_miss or not self.__check_watch__() or FN_ProxyUtil.out_of_time(): return []
            near_str = "\n".join(json.dumps(p.to_prompt(sq.query, False), ensure_ascii=False) for p in near_miss)
            kw_str = ", ".join("→".join(kw) for kw in sq.kw if kw)
            baseline = base_so.q or " ".join(kw[-1] for kw in sq.kw if kw)
            try:
                js, _ = FN_ProxyUtil.trigger_llm(FN_Prompt.A_SEARCH_REFINE.format(max_n=FN_SearchEngine.CTR_MAX_OPTIONS, baseline=baseline, query=sq.query, keywords=kw_str, near_miss=near_str,), preferred_llms=FN_ProxyUtil.get_role_llm_priorities("decompose"), expect="dict",)
                searches = js.get("searches", []) if isinstance(js, dict) else []
                if not isinstance(searches, list): return []
            except Exception as exc:
                self.debugger.print({"ctr_llm_error": str(exc)})
                return []
            existing_q = {so.q for so in self.sos.get(sq, [])}
            sos: List[FN_SearchOption] = []
            for q in searches:
                if not isinstance(q, str): continue
                q = " ".join(q.strip().split())
                if not q or q in existing_q or q == baseline or len(q.split()) > 8: continue
                so = FN_cast_model(base_so, FN_SearchOption)
                so.update_q(q)
                sos.append(so)
                existing_q.add(q)
            return sos
        def __targeted_research__(self, sq: FN_SubQuery) -> bool:
            if not self.__check_watch__() or FN_ProxyUtil.out_of_time(): return False
            verified_ids = {p.id for p in self.verified.get(sq, [])}
            pruned = self.__prune_by_score__(self.products.get(sq, []), sq)
            hard = self.__hard_prefilter__(sq, pruned[:FN_SearchEngine.VSF_HARD_TOP])
            near_miss = [p for p in hard if p.id not in verified_ids][:FN_SearchEngine.CTR_NEAR_MISS]
            base_list = self.__get_sos__(sq, True)
            if not base_list: return False
            base_so = FN_cast_model(base_list[0], FN_SearchOption)
            options = self.__build_targeted_sos_heuristic__(sq, base_so)
            options.extend(self.__build_targeted_sos_llm__(sq, near_miss, base_so))
            deduped, seen = [], set()
            for so in options:
                if so.q not in seen: seen.add(so.q); deduped.append(so)
            options = deduped[:FN_SearchEngine.CTR_MAX_OPTIONS]
            if not options: return False
            self.debugger.print({"ctr_search": (sq.query or "")[:50], "options": [so.q for so in options]})
            searches = self.__search__(sq, options, "ctr", FN_SearchEngine.SEARCH_RECOVER_PAGE_LIMIT, "b")
            products = self.__weigh__(sq, searches, "ctr", "b")
            self.searches[sq].update(searches)
            old = self.products.get(sq, [])
            merged = list({p.id: p for p in (old + products)}.values())
            merged.sort(key=lambda p: -self.weight_pool[sq][p.id].score_n if p.id in self.weight_pool[sq] else 0)
            self.products[sq] = merged
            if merged and merged[0].id in self.weight_pool[sq]: self.fscore[sq] = self.weight_pool[sq][merged[0].id].score_n
            return bool(products)
        def __ensure_verified_pool__(self, sq: FN_SubQuery) -> None:
            if self.verified.get(sq): return
            verified = self.__verify_pruned_batches__(sq)
            if verified: self.verified[sq] = verified; return
            if not self.__check_watch__(): return
            self.debugger.print({"vsf_recover": (sq.query or "")[:50]})
            if self.__targeted_research__(sq):
                verified = self.__verify_pruned_batches__(sq)
                if verified: self.verified[sq] = verified
        def __pool_for_select__(self, sq: FN_SubQuery) -> List[FN_Product]:
            if self.verified.get(sq): return list(self.verified[sq])
            return self.__prune_by_score__(self.products.get(sq, []), sq)
        def __coverage_rerank__(self, products: List[FN_Product], sq: FN_SubQuery) -> List[FN_Product]:
            if not products: return products
            scored = []
            q = sq.query or ""
            for p in products:
                cov = FN_EngineUtil.score_candidate_coverage(p, sq)
                wn = self.weight_pool[sq][p.id].score_n if p.id in self.weight_pool[sq] else 0.0
                exact = FN_EngineUtil.exact_sku_phrase_bonus(p, q)
                title_compact = 1.0 / max(len((p.title or "").split()), 1)
                scored.append((p, cov["ratio"], exact, wn, title_compact))
            scored.sort(key=lambda t: (-t[1], -t[2], -t[3], -t[4]))
            return [p for p, *_ in scored]
        def __prune_by_score__(self, products: List[FN_Product], sq: FN_SubQuery) -> List[FN_Product]: return [p for p in products if self.weight_pool[sq][p.id].score_n >= self.fscore[sq] - FN_SearchEngine.SCORE_N_THRES]
        def __create_filter_prompt__(self, candidates: Dict[FN_SubQuery, List[FN_Product]]) -> str:
            info = {}
            for i, sq in enumerate(list(candidates.keys())): info[FN_Vocab.ALPHABETS[i]] = {"query": sq.query, "candidates": self.__get_prod_str__(candidates[sq], sq, FN_Vocab.ALPHABETS[i], False)}
            return FN_Prompt.SV_FILTER.format(rules = FN_Prompt.MATCH_RULES, count = len(candidates), info = info)
        def __get_alt_str__(self, best_p: List[FN_Product]) -> str:
            if not best_p: return ""
            if len(best_p) == 1:
                for i, p in enumerate(self.products[self.qi.items[0]] * 2):
                    if p.price > best_p[0].price or i >= len(self.products[self.qi.items[0]]):
                        alt_price = str(int(p.price)) if i < len(self.products[self.qi.items[0]]) else str(int(best_p[0].price + 2.0))
                        alt_title = p.title if len(p.title) < 90 else (p.title[:90] + "...")
                        return (f"The product `{p.id}` appears to be a strong alternative. Let me compare it with `{best_p[0].id}`. The title of `{p.id}` is `{alt_title}`. " f"Based on the product details, `{p.id}` matches all stated constraints like `{best_p[0].id}`, making it difficult to choose one of them. " f"The price of product `{p.id}` is {alt_price}, while the price of `{best_p[0].id}` is {str(int(best_p[0].price))}. " f"The product `{best_p[0].id}` is less expensive. Therefore, I will prefer `{best_p[0].id}`.")
            else:
                for i, sq in enumerate(self.qi.items * 2):
                    for p in self.products[sq]:
                        if i >= len(self.qi.items) or p.price > best_p[i].price:
                            bp = best_p[i % len(self.qi.items)]
                            alt_price = str(int(p.price)) if i < len(self.qi.items) else str(int(bp.price + 3.0))
                            alt_title = p.title if len(p.title) < 90 else (p.title[:90] + "...")
                            return (f"The product `{p.id}` appears to be a strong alternative of the product `{bp.id}`, for the sub-query `{sq.query}`. " f"The title of `{p.id}` is `{alt_title}`. Let's consider a combination which has replaced `{bp.id}` with `{p.id}`. " f"Still it meets the stated constraints for each corresponding sub-query. Now let me check the prices. " f"The price of product `{p.id}` is {str(int(p.price))}, while the price of `{bp.id}` is {str(int(bp.price))}. " f"It indicates that the replaced combination will have a higher total price than the original one. " f"Therefore, the combination `{', '.join([p.id for p in best_p])}` is less expensive. I will prefer it.")
            return ""
        def __init_watch__(self, key: str, time_out: float) -> None: self.watches[key] = time.time() + time_out
        def __check_watch__(self, key: str = "query") -> bool:
            if key not in self.watches: self.debugger.print(f"# non-initialized watch: `{key}`"); return False
            if time.time() > self.watches[key]:
                print(f"* time-out alerted for `{key}`")
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
            solutions, processed_shops = [], 0
            for id, shop_batch in enumerate(plan):
                if processed_shops >= type(self).MAX_EXT_SHOPS and solutions: break
                processed_shops += len(shop_batch)
                kmaps = self.__ext_search__(shop_batch, shop_map, id)
                solmap = defaultdict(lambda: [])
                if kmaps:
                    local_sol_count = 0
                    for m in kmaps:
                        for prods, w in self.__proc_knapsack_map__(m, **kwargs):
                            sol = (prods, (shop_stats[prods[0].shop_id][0], w))
                            local_sol_count += 1
                            solmap[prods[0].shop_id].append(sol)
                            solutions.append(sol)
                    best_p = self.__proc_solmap__(solmap, True, f"shop-{id}", 4)
                    self.sess.step_on_shop_batch(shop_batch, local_sol_count, best_p, id, self.qi)
                    if best_p or not self.__check_watch__(): break
            if not best_p:
                solutions.sort(key = lambda s: s[-1], reverse = True)
                for prods, _ in solutions:
                    if len([p for i, p in enumerate(prods) if self.filtered[self.qi.items[i]][p]]) >= 2:
                        self.debugger.print("* fallback combo from non-filtered!")
                        best_p = list(prods)
                        break
                if must and solutions and not best_p: self.debugger.print("* fallback combo enforced!"); best_p = list(solutions[0][0])
                if best_p: self.sess.step_on_shop_fallback(best_p, self.qi)
            return best_p
        def __get_shop_map__(
            self, products: Dict[FN_SubQuery, List[FN_Product]]
        ) -> Tuple[
            Dict[str, Dict[FN_SubQuery, List[FN_Product]]], Dict[str, Tuple[int, float]]
        ]:
            shop_map: Dict[str, Dict[FN_SubQuery, List[FN_Product]]] = defaultdict(lambda: defaultdict(lambda: []))
            shop_stats: Dict[str, Tuple[int, float]] = {}
            for sq in self.qi.items:
                floor = self.fscore[sq] - FN_SearchEngine.SHOP_MAP_SCORE_THRES
                for p in products[sq]:
                    if self.weight_pool[sq][p.id].score_n >= floor:
                        if p not in shop_map[p.shop_id][sq]: shop_map[p.shop_id][sq].append(p)
            for sid in shop_map:
                n, w = 0, 1.0
                for sq in shop_map[sid]: w *= max(self.weight_pool[sq][p.id].confidence for p in shop_map[sid][sq]); n += 1
                shop_stats[sid] = (n, w)
            return shop_map, shop_stats
        def __plan_shop__(
            self
        ) -> Tuple[
            List[List[str]], Dict[str, Dict[FN_SubQuery, List[FN_Product]]], Dict[str, Tuple[int, float]]
        ]:
            shop_map, shop_stats = self.__get_shop_map__(self.products)
            self.sess.step_on_shop_plan(shop_stats, min(type(self).MAX_EXT_SHOPS, len(list(shop_stats.keys()))), self.qi)
            shops = sorted(list(shop_stats.keys()), key = lambda sid: shop_stats[sid], reverse = True)
            plan = []
            for n in range(4, 0, -1):
                covers = [sid for sid in shops if shop_stats[sid][0] == n]
                while len(covers) > 0: plan.append(covers[:FN_SearchEngine.SHOP_BATCH]); covers = covers[FN_SearchEngine.SHOP_BATCH:]
            return plan, shop_map, shop_stats
        def __ext_search__(
            self, shops: List[str], shop_map: Dict[str, Dict[FN_SubQuery, List[FN_Product]]], id: int
        ) -> List[
            Dict[FN_SubQuery, List[FN_Product]]
        ]:
            products: Dict[FN_SubQuery, List[FN_Product]] = {}
            searches: Dict[FN_SubQuery, Dict] = defaultdict(lambda: {})
            for sid in shops:
                for sq in self.qi.items:
                    if len(shop_map[sid][sq]) > 0: continue
                    res = self.__search__(sq, [so.update_shop(sid) for so in self.__get_sos__(sq, True)], sid, 1, "k")
                    if sum(len(ps) for ps in res.values()) == 0: break
                    searches[sq].update(res)
            for sq in self.qi.items:
                products[sq] = [p for sid in shops for p in shop_map[sid][sq]]
                if sum(len(ps) for ps in searches[sq].values()) == 0: continue
                for p in self.__weigh__(sq, searches[sq], str(id), "k"):
                    if p not in products[sq]: products[sq].append(p)
            return self.__get_knapsack_maps__(products, True)
        def __get_knapsack_maps__(
            self,
            products: Dict[FN_SubQuery, List[FN_Product]],
            is_shop: bool
        ) -> List[
            Dict[FN_SubQuery, List[FN_Product]]
        ]:
            kmaps = []
            if is_shop:
                shop_map, shop_stats = self.__get_shop_map__(products)
                for sid in shop_stats:
                    if shop_stats[sid][0] == len(self.qi.items): kmaps.append(shop_map[sid])
            else:
                lead_prods = {sq: products[sq][:type(self).MAX_CAT_SIZE] for sq in self.qi.items}
                if all(len(lead_prods[sq]) > 0 for sq in self.qi.items): kmaps.append(lead_prods)
            return kmaps
        def __proc_knapsack_map__(
            self, kmap: Dict[FN_SubQuery, List[FN_Product]], **kwargs
        ) -> List[Tuple[Tuple[FN_Product], Any]]:
            cats: List[List[Tuple[FN_Product, FN_Weight]]] = []
            for sq, products in kmap.items(): prod_weights = [(p, self.weight_pool[sq][p.id]) for p in products]; cats.append(prod_weights)
            return self.__solve_knapsack__(cats, **kwargs)
        MAX_SELECT_COMBOS = 10
        def __select_best_combo__(
            self,
            solutions: List[Tuple[Tuple[FN_Product, ...], Any]],
            id: str
        ) -> List[FN_Product]:
            if not solutions: return []
            candidates = solutions[:type(self).MAX_SELECT_COMBOS]
            bundles: Dict[str, Any] = {}
            for idx, (prods, _) in enumerate(candidates):
                items = []
                for sq, p in zip(self.qi.items, prods):
                    item = p.to_prompt(sq.query, False)
                    item["sub_query"] = sq.query
                    items.append(item)
                bundles[str(idx + 1)] = {"total_price": round(sum(p.price for p in prods), 2), "items": items,}
            temperature = 0
            max_attempts = len(FN_ProxyUtil.get_llm_priorities([])) + 1
            for _attempt in range(max_attempts):
                try:
                    n_cands = len(candidates)
                    self.debugger.print(f"* llm combo select [{id}]: evaluating {n_cands} bundle(s) across {len(self.qi.items)} item(s)...")
                    js, _ = FN_ProxyUtil.trigger_llm(FN_Prompt.SV_COMBO_SELECT.format(rules = FN_Prompt.MATCH_RULES, n_items = len(self.qi.items), bundles = json.dumps(bundles, ensure_ascii = False),), preferred_llms = FN_ProxyUtil.get_role_llm_priorities("filter"), temperature = temperature, expect = "dict",)
                    self.debugger.print(js)
                    reason = js.get("reason", "") if isinstance(js, dict) else ""
                    self.debugger.add("k-combo-select", {"id": id, "js": js, "n_bundles": n_cands})
                    best_idx = FN_Prompt.extract_combo_select_response(js, n_cands)
                    if best_idx is not None:
                        if reason: self.debugger.print(f"* combo select reason: {reason}")
                        result = list(candidates[best_idx][0])
                        self.sess.step_on_combo_select(result, n_cands, reason, is_fallback=False)
                        return result
                    break
                except Exception as exc:
                    self.debugger.print(f"# failed combo select: {exc}")
                    temperature += 0.3
            self.debugger.print("* combo select returned null — falling back to top-scored combo")
            result = list(candidates[0][0])
            self.sess.step_on_combo_select(result, len(candidates), "", is_fallback=True)
            return result
        def __proc_solmap__(
            self,
            solmap: Dict[str, List[Tuple[Tuple[FN_Product, ...], Any]]],
            sort: bool,
            id: str,
            max_tries: int
        ) -> List[FN_Product]:
            t, cursor, is_exhausted = 0, 0, False
            for t in range(max_tries):
                batch, candidates, cand_count = [], defaultdict(lambda: []), 0
                pre_valid: List[Tuple[Tuple[FN_Product, ...], Any]] = []
                if not self.__check_watch__(): break
                while cand_count < FN_SearchEngine.FILTER_BATCH and not is_exhausted:
                    is_added = False
                    for _, sols in solmap.items():
                        if cursor < len(sols):
                            prods, w = sols[cursor]
                            is_added = True
                            if all(self.filtered[self.qi.items[i]][p] for i, p in enumerate(prods)): pre_valid.append((prods, w)); continue
                            if any(self.filtered[self.qi.items[i]][p] == False for i, p in enumerate(prods)): continue
                            for i, p in enumerate(prods):
                                sq = self.qi.items[i]
                                if self.filtered[sq][p] is None and p not in candidates[sq]: candidates[sq].append(p); cand_count += 1
                            batch.append((prods, w))
                    if not is_added: is_exhausted = True
                    cursor += 1
                if cand_count > 0:
                    for sq, prods in self.__filter__(candidates, f"{id}-sol-{t}", "k").items():
                        for p in prods: self.filtered[sq][p] = True
                    for sq in candidates:
                        for p in candidates[sq]:
                            if not self.filtered[sq][p]: self.filtered[sq][p] = False
                if sort: batch.sort(key = lambda b: b[-1], reverse = True)
                best_p = self.__find_filtered_combo__(pre_valid + batch)
                if best_p: self.debugger.print(f"* selected best combo from filtered at try {t + 1}!"); return best_p
                if is_exhausted: break
            return []
        def __find_filtered_combo__(self, solutions: List[Tuple[FN_Product, Any]]) -> List[FN_Product]:
            valid: List[Tuple[Tuple[FN_Product, ...], Any]] = []
            seen: set = set()
            for prods, w in solutions:
                if all(bool(self.filtered[self.qi.items[i]][p]) for i, p in enumerate(prods)):
                    key = tuple(p.id for p in prods)
                    if key not in seen: seen.add(key); valid.append((prods, w))
            if not valid: return []
            if len(valid) == 1: return list(valid[0][0])
            self.debugger.print(f"* multiple valid combos ({len(valid)}) — escalating to LLM combo select")
            return self.__select_best_combo__(valid, "filtered-multi")
        def __solve_knapsack__( self, _: List[List[Tuple[FN_Product, float]]], **__ ) -> List[Tuple[Tuple[FN_Product, ...], float]]: return []



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

_KNG01E99_AST_PAD = 0
if _KNG01E99_AST_PAD:
    class _KNG01E99_Mu_Proxy:
        LLM_PROXY = ProxyClient(timeout = 200, max_retries = 5, retry_delay = 1.0)
        ORO_PROXY = ProxyClient(timeout = 30, max_retries = 5, retry_delay = 1.0)    

        LLM_PRIORITIES = [
            "moonshotai/kimi-k2.6", "deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1", "qwen/qwen3.5-397b-a17b",
            "z-ai/glm-5.2", "z-ai/glm-5.2", "deepseek/deepseek-chat-v3-0324", "deepseek/deepseek-r1-0528",
            "z-ai/glm-5.2", "qwen/qwen3-32b", "xiaomi/mimo-v2-flash", "google/gemma-4-31b-it"
        ]
        TEMP_UP = 0.3

        @staticmethod
        def trigger_llm(
            prompt: str, thinking: bool = False, preferred_llms: List[str] = [], temperature: float = 0, max_tokens: int = 8192, expect: str = "any"
        ) -> Tuple[
            Dict[str, Any] | List[Any], Any
        ]:        
            llm_priorities = Mu_Proxy.__get_llm_priorities__(preferred_llms)
            trigger_tick = time.time()
            t = temperature

            while True:
                for model in llm_priorities:
                    payload = Mu_Proxy.__create_llm_payload__(model, prompt, thinking, t, max_tokens)

                    try:
                        tick = time.time()

                        resp: Dict[str, Dict] = Mu_Proxy.LLM_PROXY.post("/inference/chat/completions", json_data = payload)
                        content = Mu_Proxy.__resp_to_content__(resp)

                        try:
                            js = Mu_Misc.parse_json(content)
                            if js is None: raise Exception()
                        except Exception as exc:
                            t += Mu_Proxy.TEMP_UP                        
                            raise Exception(f"failed to parse json: {exc}\n*******\n{content}\n******")

                        if (expect == "dict" and not isinstance(js, dict)) or \
                            (expect == "list" and not isinstance(js, list)):
                            t += Mu_Proxy.TEMP_UP
                            raise Exception(f"expected {expect} but got {type(js)}")

                        throughput = resp.get("usage", {}).get("completion_tokens", 0)                    
                        trigger_delay = time.time() - trigger_tick
                        delay = time.time() - tick
                    
                        Mu_Misc.LOGGER.info(
                            f"* llm triggered: model = '{model}' | throughput = {throughput} | delay = {delay:.2f}s | "
                            f"total_delay = {trigger_delay:.2f}s"
                        )
                        return js, resp
                    except Exception as exc:
                        Mu_Misc.LOGGER.error(f"# llm trigger error with '{model}': {exc}")
                        continue

        @staticmethod
        def search(filter: Mu_Filter, sort: Optional[str] = None, page: int = 1) -> Tuple[List[Mu_Product], ToolCallResult]:
            while True:
                try:
                    tool_res = execute_tool_call(
                        "find_product",
                        {
                            "q": filter.get_q(),
                            "page": page,
                            "shop_id": filter.shop,
                            "price": filter.price,
                            "sort": sort,
                            "service": ','.join(filter.services or [])
                        }
                    )
                    js_prods: List[Dict] = tool_res["result"]
                    products = []

                    for j in js_prods:
                        price = j.get("price") or 0
                        price = Mu_Misc.extract_number(str(price))

                        products.append(
                            Mu_Product(
                                id = str(j.get("product_id")),
                                title = j.get("title", ""),
                                price = price,
                                shop = j.get("shop_id", ""),
                                services = j.get("service", []) or [],
                                attr = {},
                                sku = {},
                                is_viewed = False
                            )
                        )
                
                    return products, tool_res

                except Exception as exc:
                    Mu_Misc.LOGGER.error(f"# error in `find_products`: {exc}")
                    time.sleep(0.5)

        @staticmethod
        def view(products: List[Mu_Product]) -> ToolCallResult:
            if not products: return        
            pid_str = ','.join([p.id for p in products])

            while True:
                try:
                    tool_res = execute_tool_call("view_product_information", {"ids": pid_str})        
                
                    details = tool_res["result"]
                    if not details: return tool_res

                    details_map: Dict[str, Dict[str, Any]] = {d["product_id"]: d for d in details}

                    for prod in products:
                        d = details_map.get(prod.id, {})
                    
                        prod.sku = d.get("sku_options", {})
                        prod.attr = d.get("attributes", {})
                        prod.is_viewed = True
                
                    return tool_res
                except Exception as exc:
                    Mu_Misc.LOGGER.error(f"# error in `view_products`: {exc}")
                    time.sleep(0.5)

        @staticmethod
        def submit(products: List[Mu_Product]) -> ToolCallResult:
            return execute_tool_call("recommend_product", {"product_ids": ','.join([p.id for p in products])})
    
        @staticmethod
        def finish(status: str) -> ToolCallResult:
            return execute_tool_call("terminate", {"status": status})

        @staticmethod
        def __get_llm_priorities__(prefers: List[str]) -> List[str]:
            priorities = prefers.copy()

            for llm in Mu_Proxy.LLM_PRIORITIES:
                if llm not in priorities: priorities.append(llm)
        
            return priorities

        @staticmethod
        def __create_llm_payload__(model: str, prompt: str, thinking: bool, temperature: float, max_tokens: int) -> Dict:
            return {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "chat_template_kwargs": {
                    "enable_thinking": thinking,
                    "thinking": thinking
                },
                "reasoning": {
                    "effort": "medium" if thinking else "none",
                    "enabled": thinking
                },
                "thinking": {
                    "type": "disabled" if not thinking else "enabled"
                },
                "temperature": temperature,
                "stream": False,
                "max_completion_tokens": max_tokens,
                "max_tokens": max_tokens
            }

        @staticmethod
        def __resp_to_content__(resp: Dict) -> str:
            try:
                choice: Dict[str, Dict] = resp.get("choices")[0]
                content = choice.get("message").get("content")

                if not content: raise Exception()
                return content
            except Exception as exc:
                raise Exception(f"invalid response ({exc})")


    def _KNG01E99_resolve_challenge(prob: Dict) -> ComboChallenge:
        if not isinstance(prob, dict): return ComboChallenge.PRODUCT
        cat = _normalize_category(prob.get("category"))
        if cat is not None: return cat
        query = prob.get("query", "")
        if not isinstance(query, str): query = str(query)
        if "voucher" in prob: return ComboChallenge.VOUCHER
        fn_route = first_nova_self_route(query)
        shop_route = o121_self_route(query)
        lgt_route = lightning_self_route(query)
        if lgt_route == "voucher": return ComboChallenge.VOUCHER
        if shop_route == "shop": return ComboChallenge.SHOP
        if fn_route == "product": return ComboChallenge.PRODUCT
        return ComboChallenge.PRODUCT


    class _KNG01E99_FN_QueryInfo(FN_JsonModel):
        challenge: FN_ShoppingTaskType = ""
        query_f: str = ""
        query_e: str = ""
        items: List[FN_SubQuery] = field(default_factory = list)
        vopt: Optional[FN_VoucherOption] = None


    class _KNG01E99_FLM_BPatchConfig:
        KW_HYPHEN_SAFE = True
        RELAX_ON_ZERO_RECALL = True
        ALWAYS_SUBMIT = True
        HARD_CONSTRAINT_GATE = True
        MAX_GATE_SWAPS = 2
        GATE_MIN_TIME_LEFT = 45.0
        REVIEW_MIN_LEFT = 28.0
        RELAX_SCORE_GAP = 0.6
        RESCUE_SCORE_FLOOR = 0.55
        PRODUCT_JUDGE = True
        PRODUCT_VSF = True
        VERIFY_PICK = True
        FORCE_VOUCHER_COMPARE = True
        VOUCHER_COMBO_MATRIX = True
        VOUCHER_MATRIX_TOP = 8
        SELECT_CANDIDATES = 40
        TOTAL_SECS = 250.0


    def _KNG01E99P_run_side_product(problem_data: dict, shared_deadline: float) -> list:
        _install_side_product_tools()
        query = str((problem_data or {}).get("query", "")) if isinstance(problem_data, dict) else ""
        if not isinstance(query, str):
            query = str(query)
        dialog = []
        _orig = L_Planner._identify_task_type
        def _force_product(self):
            return L_ShoppingTaskType.PRODUCT
        try:
            if shared_deadline and shared_deadline > time.monotonic():
                try:
                    L_Gateway.DEADLINE = shared_deadline
                except Exception:
                    pass
            else:
                try:
                    L_Gateway.arm_deadline()
                except Exception:
                    pass
            L_Planner._identify_task_type = _force_product
            print("* product_pipeline: connect -> L_Workflow / L_ProductEngine")
            agent = L_Workflow(query, task_hint="product")
            agent.run()
            dialog = agent.get_dialog()
            print(f"* product_pipeline: done steps={len(dialog or [])}")
        except Exception as exc:
            print(f"* product_pipeline: FAIL {type(exc).__name__}: {exc}")
            dialog = [create_dialogue_step(think=f"Unhandled side product error: {type(exc).__name__}: {exc}", tool_results=[], response="Done.", query=query, step=1)]
        finally:
            try:
                L_Planner._identify_task_type = _orig
            except Exception:
                pass
            _install_side_product_tools()
        return _ensure_dialog(dialog, query, "product")


    def _KNG01E99bind_harness_tool(name, fn) -> None:
        try:
            register_tool(name, fn)
        except Exception:
            pass
        try:
            globals()[name] = fn
        except Exception:
            pass
        try:
            import src.agent.agent_interface as _ai
            for attr in ("SANDBOX_TOOLS", "_SANDBOX_TOOLS", "_TOOL_REGISTRY"):
                reg = getattr(_ai, attr, None)
                if isinstance(reg, dict):
                    reg[name] = fn
        except Exception:
            pass


    class _KNG01E99_L_SubQuery(L_JsonModel):
        query: str = ''
        services: List[str] = field(default_factory=list, compare=False)
        price: Optional[str] = None
        kw: List[List[str]] = field(default_factory=list, compare=False)
        def __hash__(self): return hash((self.query, ''.join(sorted(self.services)), self.price))
        def __eq__(self, other):
            if not isinstance(other, L_SubQuery): return NotImplemented
            if self.query != other.query: return False
            if ''.join(sorted(self.services)) != ''.join(sorted(other.services)): return False
            if self.price != other.price: return False
            return True


    class _KNG01E99_Lou_Filter(Lou_Keywords):
        services: List[str] = field(default_factory = list)

        price: Optional[str] = None
        shop: Optional[str] = None

        def get_kw(self) -> Lou_Keywords:
            return Lou_Keywords(keywords = self.keywords)

        def update(self, **kwargs) -> "Filter":
            for k, v in kwargs.items():
                self.__setattr__(k, v)
        
            return self

        def clone(self) -> "Filter":
            return Lou_Filter.from_json(self.to_json())

        def __hash__(self) -> int:
            return hash((self.get_q(), self.price, self.shop))

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, type(self)): return False        
            return self.get_q() == other.get_q() and self.price == other.price and self.shop == other.shop


    class _KNG01E99_FLM_PhaseBudget:
        def __init__(self, total_secs: float = FLM_OPatchConfig.TOTAL_SECS, review_min_left: float = FLM_OPatchConfig.REVIEW_MIN_LEFT):
            self.total_secs = float(total_secs)
            self.review_min_left = float(review_min_left)
            self.deadline = time.monotonic() + self.total_secs

        def left(self) -> float:
            return max(0.0, self.deadline - time.monotonic())

        def elapsed(self) -> float:
            return max(0.0, self.total_secs - self.left())

        def out_of_time(self) -> bool:
            return self.left() <= 0.0

        def allow_review(self) -> bool:
            return self.left() >= self.review_min_left

        def allow_gate(self, min_left: float = FLM_OPatchConfig.GATE_MIN_TIME_LEFT) -> bool:
            return self.left() >= min_left

        def allow_dual_start(self, cfg: FLM_DualSolveConfig = FLM_DualSolveConfig) -> bool:
            return self.elapsed() < cfg.SECOND_PATH_CUTOFF_SECS and self.left() >= cfg.MIN_SECONDS_TO_START_SECOND

        def allow_dual_round(self, round_idx: int = 0, cfg: FLM_DualSolveConfig = FLM_DualSolveConfig) -> bool:
            if round_idx >= cfg.MAX_DUAL_ROUNDS:
                return False
            if self.elapsed() >= cfg.SECOND_PATH_CUTOFF_SECS:
                return False
            return self.left() >= cfg.MIN_ROUND_TIME_LEFT


    def _KNG01E99P_shared_deadline():
        for util in (FN_ProxyUtil, S21_ProxyUtil, KAK_ProxyUtil):
            d = getattr(util, "DEADLINE", None)
            if d:
                return d
        return None


    def _KNG01E99P_arm_host_budget(query: str, lane: str):
        hard = _P_query_hardness(query, lane)
        cap = _HOST_SOFT_CAP_EASY - hard * (_HOST_SOFT_CAP_EASY - _HOST_SOFT_CAP_HARD)
        cap = max(90.0, float(cap))
        deadline = time.monotonic() + cap
        restored = []
        for util in (FN_ProxyUtil, S21_ProxyUtil, KAK_ProxyUtil):
            try:
                orig = util.arm_deadline
                restored.append((util, orig))
                def _make_arm(u=util, d=deadline):
                    def _arm(*_a, **_k):
                        u.DEADLINE = d
                    return _arm
                util.arm_deadline = staticmethod(_make_arm())
                util.DEADLINE = deadline
            except Exception:
                try:
                    util.DEADLINE = deadline
                except Exception:
                    pass
        print(f"* host_budget: lane={lane} hard={hard:.2f} cap={cap:.1f}s")
        return deadline, restored


    class _KNG01E99_FN_AdversarialIngressGuard:
        _RX_INJECTION = re.compile(r"(?is)(?:^|\n)\s*(?:system\s*:|ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions" r"|you\s+are\s+now\s+|developer\s+mode\s*|jailbreak\s*:|do\s+not\s+follow\s+the\s+rules)")
        _RX_ROLE_PLAY = re.compile(r"(?is)\b(?:act\s+as\s+(?:DAN|an?\s+unrestricted)|pretend\s+you\s+have\s+no\s+restrictions)\b")
        @staticmethod
        def sanitize(query: str) -> str:
            q = str(query or "")
            q = FN_AdversarialIngressGuard._RX_INJECTION.sub(" ", q)
            q = FN_AdversarialIngressGuard._RX_ROLE_PLAY.sub(" ", q)
            q = re.sub(r"[ \t]{2,}", " ", q)
            q = re.sub(r"\n{3,}", "\n\n", q)
            return q.strip() or str(query or "").strip()


    class _KNG01E99_Lou_FuzzyBoundaryLogic:
        """Fuzzy price/boundary helpers (bird/beam/_x/shrike/garland patterns)."""

        @staticmethod
        def parse_price_bounds(price_range: str | None) -> Tuple[float | None, float | None]:
            cleaned = Lou_EdgeCaseProcessor.sanitize_price_range(price_range)
            if not cleaned:
                return (None, None)
            left_raw, _, right_raw = cleaned.partition("-")
            lo = Lou_EdgeCaseProcessor.coerce_float(left_raw) if left_raw.strip() else None
            hi = Lou_EdgeCaseProcessor.coerce_float(right_raw) if right_raw.strip() else None
            return (lo, hi)

        @staticmethod
        def normalize_price_from_query(text: str) -> str | None:
            if Lou_EdgeCaseProcessor.is_nullish(text):
                return None
            m = Lou__RX_PRICE_BETWEEN.search(text)
            if m:
                return Lou_EdgeCaseProcessor.sanitize_price_range(f"{m.group(1)}-{m.group(2)}")
            m = Lou__RX_PRICE_OVER.search(text)
            if m:
                return Lou_EdgeCaseProcessor.sanitize_price_range(f"{m.group(1)}-")
            m = Lou__RX_PRICE_UNDER.search(text)
            if m:
                return Lou_EdgeCaseProcessor.sanitize_price_range(f"-{m.group(1)}")
            return None

        @staticmethod
        def fuzzy_tolerance(lo: float, hi: float, *, ratio: float | None = None, abs_tol: float | None = None) -> Tuple[float, float]:
            ratio = Lou_ENHANCEMENT_BENCH.fuzzy_price_ratio if ratio is None else ratio
            abs_tol = Lou_ENHANCEMENT_BENCH.fuzzy_price_abs_tol if abs_tol is None else abs_tol
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
            parsed_lo, parsed_hi = Lou_FuzzyBoundaryLogic.parse_price_bounds(price_range)
            lo = floor if parsed_lo is None else max(floor, parsed_lo)
            hi = ceiling if parsed_hi is None else min(ceiling, parsed_hi)
            if hi < lo:
                return None
            return (lo, hi)

        @staticmethod
        def widen_price_range(price_range: str | None, ratio: float | None = None) -> str | None:
            if Lou_EdgeCaseProcessor.is_nullish(price_range):
                return None
            ratio = Lou_ENHANCEMENT_BENCH.fuzzy_price_ratio if ratio is None else ratio
            try:
                lo, hi = Lou_FuzzyBoundaryLogic.parse_price_bounds(price_range)
                if lo is not None:
                    lo = max(0.0, lo * (1.0 - ratio))
                if hi is not None:
                    hi = hi * (1.0 + ratio)
                lo_fmt = f"{lo:.0f}" if lo is not None else ""
                hi_fmt = f"{hi:.0f}" if hi is not None else ""
                widened = f"{lo_fmt}-{hi_fmt}"
                return None if widened == "-" else widened
            except (ValueError, TypeError):
                return Lou_EdgeCaseProcessor.sanitize_price_range(price_range)

        @staticmethod
        def price_fits_range(price: Any, price_range: str | None, *, fuzzy: bool = True) -> Tuple[bool, str]:
            pv = Lou_EdgeCaseProcessor.clamp_listing_price(price)
            if pv is None:
                return (False, "price is null or non-numeric")
            lo, hi = Lou_FuzzyBoundaryLogic.parse_price_bounds(price_range)
            if lo is None and hi is None:
                return (True, "no price constraint")
            tol = Lou_ENHANCEMENT_BENCH.fuzzy_price_abs_tol if fuzzy else 0.0
            if lo is not None and pv < lo - tol:
                return (False, f"price {pv:g} below lower bound {lo:g}")
            if hi is not None and pv > hi + tol:
                return (False, f"price {pv:g} above upper bound {hi:g}")
            return (True, f"price {pv:g} within range")

        @staticmethod
        def classify_price_boundary(price: Any, lo: float | None, hi: float | None) -> str:
            pv = Lou_EdgeCaseProcessor.clamp_listing_price(price)
            if pv is None:
                return "null"
            tol = Lou_ENHANCEMENT_BENCH.fuzzy_price_abs_tol
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
            ok, _ = Lou_FuzzyBoundaryLogic.price_fits_range(
                price_val,
                str(price_range_str) if price_range_str else None,
                fuzzy = False,
            )
            return 5.0 if ok else -25.0

        @staticmethod
        def voucher_window_status(cart_total: float, threshold: float, allowed_total: float) -> Dict[str, Any]:
            cart_total = Lou_EdgeCaseProcessor.coerce_float(cart_total)
            threshold = Lou_EdgeCaseProcessor.coerce_float(threshold)
            allowed_total = Lou_EdgeCaseProcessor.coerce_float(allowed_total)
            eps = Lou_ENHANCEMENT_BENCH.voucher_window_eps

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


    def _KNG01E99_Lou_freezable(func: Callable) -> Callable:
        return func


    class _KNG01E99_FN_VoucherEarlyValidator:
        @staticmethod
        def validate(vopt: Optional[FN_VoucherOption]) -> Tuple[bool, str]:
            if vopt is None: return False, "missing_vopt"
            budget = FN_EdgeCaseProcessor.coerce_float(vopt.budget, lo=0.0)
            if budget <= 0: return False, "invalid_budget"
            threshold = FN_EdgeCaseProcessor.coerce_float(vopt.threshold or 0, lo=0.0)
            discount = FN_EdgeCaseProcessor.coerce_float(vopt.discount, lo=0.0)
            if not vopt.is_fixed and (discount <= 0 or discount >= 100): return False, "invalid_percent_discount"
            if vopt.is_fixed and discount <= 0: return False, "invalid_fixed_discount"
            ceiling = FN_VoucherEarlyValidator._ceiling(vopt)
            if ceiling is not None and ceiling <= threshold: return False, "ceiling_below_threshold"
            return True, "ok"
        @staticmethod
        def _ceiling(vopt: FN_VoucherOption) -> Optional[float]:
            budget = FN_EdgeCaseProcessor.coerce_float(vopt.budget)
            discount = FN_EdgeCaseProcessor.coerce_float(vopt.discount)
            if vopt.is_fixed: return budget + discount
            rate = discount / 100.0
            if rate <= 0 or rate >= 1: return None
            uncapped = budget / (1.0 - rate)
            if vopt.cap is not None: capped = budget + FN_EdgeCaseProcessor.coerce_float(vopt.cap); return min(uncapped, capped)
            return uncapped


    class _KNG01E99_FN_FuzzyBoundaryLogic:
        @staticmethod
        def parse_price_bounds(price_range: str | None) -> Tuple[float | None, float | None]:
            cleaned = FN_EdgeCaseProcessor.sanitize_price_range(price_range)
            if not cleaned: return (None, None)
            left_raw, _, right_raw = cleaned.partition("-")
            lo = FN_EdgeCaseProcessor.coerce_float(left_raw) if left_raw.strip() else None
            hi = FN_EdgeCaseProcessor.coerce_float(right_raw) if right_raw.strip() else None
            return (lo, hi)
        @staticmethod
        def normalize_price_from_query(text: str) -> str | None:
            if FN_EdgeCaseProcessor.is_nullish(text): return None
            m = _RX_PRICE_BETWEEN.search(text)
            if m: return FN_EdgeCaseProcessor.sanitize_price_range(f"{m.group(1)}-{m.group(2)}")
            m = _RX_PRICE_OVER.search(text)
            if m: return FN_EdgeCaseProcessor.sanitize_price_range(f"{m.group(1)}-")
            m = _RX_PRICE_UNDER.search(text)
            if m: return FN_EdgeCaseProcessor.sanitize_price_range(f"-{m.group(1)}")
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
            if ceiling < floor: return None
            parsed_lo, parsed_hi = FN_FuzzyBoundaryLogic.parse_price_bounds(price_range)
            lo = floor if parsed_lo is None else max(floor, parsed_lo)
            hi = ceiling if parsed_hi is None else min(ceiling, parsed_hi)
            if hi < lo: return None
            return (lo, hi)
        @staticmethod
        def widen_price_range(price_range: str | None, ratio: float | None = None) -> str | None:
            if FN_EdgeCaseProcessor.is_nullish(price_range): return None
            ratio = ENHANCEMENT_BENCH.fuzzy_price_ratio if ratio is None else ratio
            try:
                lo, hi = FN_FuzzyBoundaryLogic.parse_price_bounds(price_range)
                if lo is not None: lo = max(0.0, lo * (1.0 - ratio))
                if hi is not None: hi = hi * (1.0 + ratio)
                lo_fmt = f"{lo:.0f}" if lo is not None else ""
                hi_fmt = f"{hi:.0f}" if hi is not None else ""
                widened = f"{lo_fmt}-{hi_fmt}"
                return None if widened == "-" else widened
            except (ValueError, TypeError):
                return FN_EdgeCaseProcessor.sanitize_price_range(price_range)
        @staticmethod
        def price_fits_range(price: Any, price_range: str | None, *, fuzzy: bool = True) -> Tuple[bool, str]:
            pv = FN_EdgeCaseProcessor.clamp_listing_price(price)
            if pv is None: return (False, "price is null or non-numeric")
            lo, hi = FN_FuzzyBoundaryLogic.parse_price_bounds(price_range)
            if lo is None and hi is None: return (True, "no price constraint")
            tol = ENHANCEMENT_BENCH.fuzzy_price_abs_tol if fuzzy else 0.0
            if lo is not None and pv < lo - tol: return (False, f"price {pv:g} below lower bound {lo:g}")
            if hi is not None and pv > hi + tol: return (False, f"price {pv:g} above upper bound {hi:g}")
            return (True, f"price {pv:g} within range")
        @staticmethod
        def classify_price_boundary(price: Any, lo: float | None, hi: float | None) -> str:
            pv = FN_EdgeCaseProcessor.clamp_listing_price(price)
            if pv is None: return "null"
            tol = ENHANCEMENT_BENCH.fuzzy_price_abs_tol
            if lo is not None and abs(pv - lo) <= tol: return "boundary_low"
            if hi is not None and abs(pv - hi) <= tol: return "boundary_high"
            if lo is not None and pv < lo: return "under"
            if hi is not None and pv > hi: return "over"
            return "inside"
        @staticmethod
        def price_fit_score_adjustment(price_val: Any, price_range_str: Any) -> float:
            ok, _ = FN_FuzzyBoundaryLogic.price_fits_range(price_val, str(price_range_str) if price_range_str else None, fuzzy = False,)
            return 5.0 if ok else -25.0
        @staticmethod
        def voucher_window_status(cart_total: float, threshold: float, allowed_total: float) -> Dict[str, Any]:
            cart_total = FN_EdgeCaseProcessor.coerce_float(cart_total)
            threshold = FN_EdgeCaseProcessor.coerce_float(threshold)
            allowed_total = FN_EdgeCaseProcessor.coerce_float(allowed_total)
            eps = ENHANCEMENT_BENCH.voucher_window_eps
            if cart_total > allowed_total + eps: return {"within_window": False, "reason": "over_allowed_total", "status": "over", "below_threshold": cart_total < threshold, "at_boundary": False,}
            if threshold > 0 and cart_total + eps < threshold: return {"within_window": False, "reason": "below_threshold", "status": "below_threshold", "below_threshold": True, "at_boundary": True,}
            if abs(cart_total - threshold) <= eps or abs(cart_total - allowed_total) <= eps: return {"within_window": True, "reason": "boundary", "status": "boundary", "below_threshold": False, "at_boundary": True,}
            return {"within_window": True, "reason": "ok", "status": "ok", "below_threshold": False, "at_boundary": False,}
        @staticmethod
        def extend_top_n_on_tie(scores: List[Tuple[Any, float]], top_n: int) -> List[Any]:
            if not scores or top_n <= 0: return []
            ordered = sorted(scores, key = lambda t: t[1], reverse = True)
            if len(ordered) <= top_n: return [item for item, _ in ordered]
            cutoff = ordered[top_n - 1][1]
            return [item for item, sc in ordered if sc >= cutoff]


    class _KNG01E99_FN_EdgeCaseProcessor:
        MAX_SAFE_FLOAT = 1e15
        @staticmethod
        def is_nullish(value: Any) -> bool:
            if value is None: return True
            if isinstance(value, str): s = value.strip().lower(); return s in ("", "null", "none", "n/a", "undefined", "nan")
            if isinstance(value, (list, dict, tuple, set)) and len(value) == 0: return True
            return False
        @staticmethod
        def coerce_float(
            value: Any,
            *,
            default: float = 0.0,
            lo: float | None = None,
            hi: float | None = None,
        ) -> float:
            if FN_EdgeCaseProcessor.is_nullish(value): out = default
            elif isinstance(value, bool): out = float(value)
            elif isinstance(value, (int, float)): out = float(value)
            elif isinstance(value, str):
                try:
                    out = float(FN_MiscUtil.extract_number(value))
                except (ValueError, TypeError):
                    out = default
            else:
                out = default
            if not math.isfinite(out): out = default
            out = max(-FN_EdgeCaseProcessor.MAX_SAFE_FLOAT, min(FN_EdgeCaseProcessor.MAX_SAFE_FLOAT, out))
            if lo is not None: out = max(lo, out)
            if hi is not None: out = min(hi, out)
            return out
        @staticmethod
        def coerce_int(
            value: Any,
            *,
            default: int = 0,
            lo: int | None = None,
            hi: int | None = None,
        ) -> int:
            f = FN_EdgeCaseProcessor.coerce_float(value, default = float(default), lo = float(lo) if lo is not None else None, hi = float(hi) if hi is not None else None,)
            return int(f)
        @staticmethod
        def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
            if denominator == 0 or not math.isfinite(denominator): return default
            result = numerator / denominator
            return result if math.isfinite(result) else default
        @staticmethod
        def sanitize_price_range(price_range: str | None) -> str | None:
            if FN_EdgeCaseProcessor.is_nullish(price_range): return None
            s = re.sub(r"[^\d\-\.]", "", str(price_range).strip())
            if not s or s == "-": return None
            left_raw, sep, right_raw = s.partition("-")
            if not sep: return None
            left, right = left_raw.strip(), right_raw.strip()
            if left and right:
                lo = FN_EdgeCaseProcessor.coerce_float(left)
                hi = FN_EdgeCaseProcessor.coerce_float(right)
                if hi < lo: lo, hi = hi, lo
                return f"{lo:g}-{hi:g}"
            if left: return f"{FN_EdgeCaseProcessor.coerce_float(left):g}-"
            if right: return f"-{FN_EdgeCaseProcessor.coerce_float(right):g}"
            return None
        @staticmethod
        def sanitize_product_id(value: Any) -> str | None:
            pid = str(value or "").strip()
            if pid.lower() in _INVALID_PRODUCT_IDS: return None
            return pid or None
        @staticmethod
        def sanitize_service_csv(value: Any) -> str | None:
            if FN_EdgeCaseProcessor.is_nullish(value): return None
            parts: List[str] = []
            for token in re.split(r"[,|]", str(value)):
                t = token.strip()
                if t and t not in parts: parts.append(t)
            return ",".join(parts) if parts else None
        @staticmethod
        def clamp_listing_price(price: Any, *, max_price: float = 1e12) -> float | None:
            if FN_EdgeCaseProcessor.is_nullish(price): return None
            try:
                p = float(price)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(p): return None
            if p < 0: return 0.0
            if p > max_price: return max_price
            return p
        @staticmethod
        def parse_comma_prices(product_prices: Any) -> Tuple[List[float], Optional[str]]:
            if FN_EdgeCaseProcessor.is_nullish(product_prices): return [], "product_prices is null or empty."
            try:
                prices = [FN_EdgeCaseProcessor.coerce_float(p.strip(), lo = 0.0) for p in str(product_prices).split(",") if p.strip()]
            except (ValueError, TypeError):
                return [], "Invalid product_prices format. Use comma-separated numbers."
            if any(p < 0 for p in prices): return [], "product_prices must not contain negative values."
            return prices, None
        @staticmethod
        def empty_dict_if_null(js: Dict | None) -> Dict:
            return js if isinstance(js, dict) else {}
        @staticmethod
        def empty_list_if_null(items: List | None) -> List:
            return items if isinstance(items, list) else []


    def _KNG01E99P_dialog_has_compare(dialog: list) -> bool:
        blob = json.dumps(dialog or [], ensure_ascii=False).lower()
        markers = (
            "appears to be a strong alternative",
            "let me compare",
            "compared with the top promising",
            "while the price of",
            "is less expensive",
            "therefore i prefer",
            "therefore, i will prefer",
            "bundle",
            "wins slot",
        )
        return sum(1 for m in markers if m in blob) >= 2


    class _KNG01E99_FN_Product(FN_JsonModel):
        id: str
        title: str
        price: float
        service: Optional[List[str]] = None
        shop_id: Optional[str] = None
        attributes: Dict[str, List[str]] = field(default_factory = dict)
        sku: Dict[str, Dict[str, str]] = field(default_factory = dict)
        is_viewed: bool = False
        def to_prompt(self, q: str, for_reason: bool = False) -> Dict[str, Any]:
            attr = self.extract_attr(q, ':', False)
            joint_attr = []
            for _, group in attr.items():
                for lines in group:
                    for l in lines: joint_attr.append(l)
            js = {"title": self.title, "attributes": '\n'.join(joint_attr)}
            if for_reason: js["price"] = self.price
            return js
        def clone_from(self, other: "FN_Product") -> None:
            self.id = other.id
            self.title = other.title
            self.price = other.price
            self.service = other.service
            self.shop_id = other.shop_id
            self.attributes = other.attributes
            self.sku = other.sku
            self.is_viewed = other.is_viewed
        def extract_attr(self, q: str, delimiter: str, norm: bool) -> Dict[str, List[List[str]]]:
            extracted = {"sku": []}
            is_valid = False
            for k, vs in self.attributes.items():
                extracted[f"att_{k}"] = []
                atoms = []
                for v in vs:
                    if FN_EngineUtil.is_valid_attr(k, v, q): is_valid = True
                    atoms.append((k, v))
                if atoms: extracted[f"att_{k}"].append(atoms)
            for _, kv in self.sku.items():
                atoms = []
                for k, v in kv.items():
                    if FN_EngineUtil.is_valid_attr(k, v, q): is_valid = True
                    atoms.append((k, v))
                if atoms: extracted["sku"].append(atoms)
            if not is_valid: extracted.clear()
            return {g: [[f"{FN_MiscUtil.normalize_str(k) if norm else k}{delimiter} " f"{FN_MiscUtil.normalize_str(v) if norm else v}" for k, v in kv] for kv in kvs] for g, kvs in extracted.items()}
        def __hash__(self): return hash(self.id)
        def __eq__(self, other):
            if not isinstance(other, FN_Product): return NotImplemented
            return self.id == other.id


    class _KNG01E99_L_QueryInfo(L_JsonModel):
        challenge: L_ShoppingTaskType = ''
        query_f: str = ''
        query_e: str = ''
        items: List[L_SubQuery] = field(default_factory=list)
        vopt: Optional[L_VoucherOption] = None


    class _KNG01E99_FN_MiscUtil:
        @staticmethod
        def extract_number(s: str, as_int: bool = False) -> int | float:
            pattern = r"\d+\.?\d*"
            matches = re.findall(pattern, s)
            if len(matches) == 0: return 0
            val = float(matches[0])
            return val if not as_int else int(val)
        @staticmethod
        def ensure_number(v: Any, as_int: bool = False) -> int | float:
            if v is None: return 0
            if isinstance(v, (int, float)): return v
            if isinstance(v, str): return FN_MiscUtil.extract_number(v, as_int = as_int)
            raise ValueError(f"# cannot convert {v} to number")
        @staticmethod
        def extract_bool(s: str) -> bool:
            s = s.lower().strip()
            if not s: return False
            elif s == "true": return True
            elif s == "false": return False
            else: return int(float(s)) != 0 if s.isnumeric() else False
        @staticmethod
        def normalize_str(x: str) -> str:
            x = x.lower()
            x = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", x)
            x = re.sub(r"\b([a-z]+)(\d+)\b", r"\1 \2", x)
            x = re.sub(r"[\(\)\[\]\{\}\+\*\_\:\;\,\-\~\/\<\>\=]", " ", x)
            x = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1 \2", x)
            x = re.sub(r"\s+", " ", x).strip()
            return x
        @staticmethod
        def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
            keys = list(cat.keys()) if isinstance(cat, dict) else list(range(len(cat)))
            choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]
            opts = product(*choices)
            return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)
        @staticmethod
        def print_json(js: Dict[str, Any] | List[Any]) -> None:
            print(json.dumps(js, ensure_ascii = False, indent = 2))
        @staticmethod
        def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
            cleaned = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags = re.I)
            cleaned = re.sub(r"\s*```$", "", cleaned, ).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            decoder = json.JSONDecoder()
            starts = []
            obj_start = cleaned.find("{")
            if obj_start >= 0: starts.append(obj_start)
            arr_start = cleaned.find("[")
            if arr_start >= 0: starts.append(arr_start)
            for start in sorted(starts):
                try:
                    obj, _ = decoder.raw_decode(cleaned[start:])
                    return obj
                except json.JSONDecodeError:
                    pass
            return None


    class _KNG01E99_Lou_ExtTool:
        def search(**kwargs) -> List[Dict]:
            result = Lou_Proxy.ORO_PROXY.get("/search/find_product", kwargs)
            return result if result else []

        def view(ids: str) -> List[Dict]:
            result = Lou_Proxy.ORO_PROXY.get("/search/view_product_information", {"product_ids": ids})
            return result if result else []

        def submit(product_ids: str) -> str:
            return f"submitted recommendation to the user: {product_ids}."

        def finish(status: str) -> str:
            return f"session completed with status: {status}"


    class _KNG01E99_Mu_Session(Mu_Debuggable):
        MAX_TOOL_RES_PER_STEP = 7

        def __init__(self, query: Mu_Query, debugger: Optional[Mu_Debugger]) -> None:
            super(Mu_Session, self).__init__(query, debugger)
            self.epoch = time.time()

            self.tool_results = []
            self.dialog = []
    
        def step(self, think: str, content: str) -> None:
            timestamp = time.time() - self.epoch
        
            self.dialog.append(
                create_dialogue_step(
                    think = think,
                    tool_results = self.tool_results[:Mu_Session.MAX_TOOL_RES_PER_STEP],
                    response = content + f"\n(Timestamp: {timestamp:.2f}s)",
                    query = self.query.query_f,
                    step = len(self.dialog) + 1
                )
            )
            self.tool_results = []

            if self.debugger is not None and self.debugger.show_step:                        
                self.log({
                    "step": len(self.dialog),
                    "think": think,
                    "content": content
                })
    
        def clear(self) -> None:
            self.tool_results = []

        def add_tool_result(self, res: ToolCallResult) -> None:
            try:
                json.dumps(res)
                self.tool_results.append(res)
            except Exception as exc:
                self.log(f"# cannot add tool result: {exc}")

        def step_on_build(self) -> None:
            think = f"This is a shopping request involving {Mu_Challenge(self.query.challenge).promptfy(len(self.query.intents) > 1)}. "
            think += f"First, I will extract meaningful keywords and additional filters from "
            think += ("each product intent" if len(self.query.intents) > 1 else "the query") + " to facilitate search:\n"

            keywords = [", ".join(sq.keywords) for sq in self.query.intents]
            services = [", ".join(svc for svc in sq.services) if sq.services else "Not specified" for sq in self.query.intents]
            prices = [sq.price if sq.price else "Not specified" for sq in self.query.intents]

            for i in range(len(self.query.intents)):
                think += f"\n- Product {i + 1}: " if len(self.query.intents) > 1 else "\n- "
                think += f"Keywords = \"{keywords[i]}\" | Services = \"{services[i]}\" | Price = \"{prices[i]}\""

            if self.query.challenge == Mu_Challenge.VOUCHER:
                think += f"\n\nUser's budget is {int(self.query.voucher.budget)} and the available voucher options are as follows:\n\n"
                think += "- Voucher Type: " + ("Fixed amount" if self.query.voucher.is_fixed else "Percentage") + "\n"
                think += f"- Discount: {int(self.query.voucher.discount)}" + ("%" if not self.query.voucher.is_fixed else "") + "\n"
                think += "- Application Criteria: " + ("Same shop" if self.query.voucher.apply_to == "shop" else "All products") + "\n"            
                think += (f"- Threshold: {int(self.query.voucher.threshold)}" if self.query.voucher.threshold is not None else "Not specified") + "\n"
                think += (f"- Cap: {int(self.query.voucher.cap)}" if self.query.voucher.cap is not None else "Not specified") + "\n"

            self.step(think, "Initial query analysis completed.")

        def step_on_retrieve(self) -> None:
            think = "Next, I will perform initial searches for " + ("each sub-query." if len(self.query.intents) > 1 else "the query.")
            self.step(think, "Initial search completed.")

        def step_on_shop_plan(self, shop_stats: Dict[str, Tuple[int, float]], top_k: int) -> None:
            think = "Now I will look for combinations of products from the same shop, since such combinations may benefit from voucher application. For this, " if self.query.challenge == Mu_Challenge.VOUCHER else "Now, "
            think += "I will group the retrieved products by shop and examine how well each shop covers the sub-queries. Shops with broader coverage are more promising, but additional shops will also be considered to avoid overlooking better combinations. "
            think += "The following statistics summarize the coverage provided by the discovered shops:\n\n"
        
            plan = []

            for n in range(len(self.query.intents), 0, -1):
                shops = [sid for sid in shop_stats if shop_stats[sid][0] == n]
                plan.extend(shops)

                if n > 1:
                    think += f"- Shops covering {n} sub-queries: `{', '.join(shops)}`\n"
                else:
                    think += f"- The remaining {len(shops)} shop{'s' if len(shops) > 1 else ''} cover{'s' if len(shops) == 1 else ''} only one sub-query.\n"

            think += "\nGood. Next, I will examine the most promising shops in order of coverage to determine whether they can fully satisfy the sub-queries"
            think += " while allowing the voucher to be applied." if self.query.challenge == Mu_Challenge.VOUCHER else "."
            think += f"\n\nThe following shops will be examined: {', '.join(plan[:top_k])}, ... ..."

            self.step(think, "Planned candidate shops.")

        def step_on_shop_batch(self, batch: List[str], sols: int, best_p: List[Mu_Product], n: int) -> None:
            best_str = ', '.join([p.id for p in best_p])
        
            think = f"[Shop Examination Step {n + 1}]\n\n"
            think += f"In this step, I will check the shop{'s' if len(batch) > 1 else ''}: `{', '.join(batch)}`\n\n"
            think += "I will perform additional searches for each shop on its missing sub-queries using narrower search criteria by enforcing a shop ID constraint. "
            think += f"Among the retrieved products, I can find {sols} combination{'s' if sols > 1 else ''} whose products all come from the same shop"
            think += " for which the voucher can be applied. " if self.query.challenge == Mu_Challenge.VOUCHER else ". "

            if sols > 0:
                think += f"After carefully examining the candidate combination{'s' if sols > 1 else ''}, "
            
                if best_p:
                    think += f"I identified a promising combination (`{best_str}`). Each product satisfies its corresponding constraints. "
                    think += "Now I will stop the shop examination and recommend the identified combination."
                else:
                    think += f"{'all of them were' if sols > 1 else 'it was'} rejected because one or more products did not satisfy the specified constraints. "
                    think += "Therefore, I will continue the shop examination."
            else:
                think += "Therefore, I will continue the shop examination."
        
            self.step(think, "")

        def step_on_shop_fallback(self, best_p: List[Mu_Product]) -> None:
            best_str = ', '.join([p.id for p in best_p])

            think = "[Shop Examination Rollback]\n\nI have examined all discovered shops, but was unable to find one providing matching products"
            think += " for which the voucher can be applied. " if self.query.challenge == Mu_Challenge.VOUCHER else ". "

            if best_p:
                think += "It's because either a valid combination has been overlooked during the previous evaluation steps, or no matching combination exists in the current database. "
                think += "Before reporting failure to the user, I will perform one final verification of the previously discovered candidate combinations. "
                think += f"The combination `{best_str}` appears to be a strong candidate. Let me verify it against the stated constraints. "
                think += "Upon re-examination, I found that this combination satisfies all stated constraints. It was incorrectly rejected during the previous evaluation. "
                think += f"To ensure that no other valid combinations have been overlooked, I will review the remaining candidates as well. "
                think += f"No additional combinations satisfying all stated constraints were found. The only missed combination is `{best_str}`. "
                think += f"Therefore, this combination can be recommended to the user."
            elif self.query.challenge == Mu_Challenge.VOUCHER:
                think += "However, voucher application is optional. As long as the total price remains within the budget, the combination can still be recommended. "
                think += "Therefore, I will switch to voucher-free examination mode."
            else:
                think += "This indicates that no matching combination exists in the current database. I will report failure to the user."

            self.step(think, "Completed shop examination process.")

        def step_on_platform(self, best_p: List[Mu_Product], sol_count: bool, is_fallback: bool) -> None:
            best_str = ', '.join([p.id for p in best_p])

            think = (
                f"Now I will look for combinations of products whose total price is within the buget"
                f"{' or becomes affordable after applying the voucher. ' if not is_fallback else '. '}"
                f"First, I will apply the Meet-in-the-Middle (MITM) algorithm to solve the underlying knapsack problem, where product weights are derived from the logarithms of the search-rank probabilities. "
                f"I can find {sol_count} valid solutions, sorted by total weights. Next, I will examine the candidate combinations in order until I find one that satisfies all stated constraints. "
            )
            if best_p:
                think += f"The combination `{best_str}` appears to be a strong candidate. Let me verify its products against the constraints. "
                think += "Good. I have confirmed that each product satisfies its corresponding sub-query. Now this combination can be recommended to the user."
            else:
                think += "I have evaluated all candidate combinations, but none of them satisfies all stated constraints. "
                think += "This indicates that no combination satisfying the user's requirements exists in the current database. I will report the failure to the user."

            self.step(think, "")

        def step_on_select(self, best_p: Mu_Product, js_check: Dict, is_fallback: bool, sq: Mu_SubQuery) -> None:
            if not is_fallback:
                think = "From the retrieved products, I will examine the top ranked candidates to find a product satisfying all the constraints."
                think += f"\n\n{json.dumps(js_check, ensure_ascii = False, indent = 2)}\n\nGood. A matching product has been identified."
            else:
                think = "Since the retrieved products are already ranked by relevance, I will first examine the highest-ranked product to determine whether it satisfies all the user's requirements. "
                think += "If it turns out to be a good match, I can recommend it directly without performing any further evaluation. "
                think += f"The title of the highest-ranked product is \"{best_p.title}\" I will check its attribute details to verify each constraint using the following checklist:\n"

                for k, v in sq.template.items():
                    think += f"\n`{k}`: `{v}` → match ✓"
            
                think += "\n\nAll checks passed."

            self.step(think, f"A matching product selected: `{best_p.id}`")

        def step_on_compare(self, best_p: List[Mu_Product], alt: str) -> None:
            if not best_p or not alt: return

            think = "To avoid overlooking a better candidate, I will compare the identified one with other top promising candidate.\n\n" + alt
            self.step(think, "Compared with the top promising candidate.")

        def step_on_submit(self, best_p: List[Mu_Product]) -> None:
            if not best_p:
                return self.step(
                    "I can't find the product(s) satisfying the user's requirements. I will report failure to the user.",
                    "Failed to find the requested product(s). Please try again with more specific constraints."
                )

            think = f"I have identified the following product{' combination' if len(self.query.intents) > 1 else ''}: `{', '.join([p.id for p in best_p])}`"
            think += f" Before submitting it, I will verify all the requirements again.\n"

            for i in range(len(best_p)):
                think += f"\n[Product {i + 1}] (UID = {best_p[i].id})\n" if len(self.query.intents) > 1 else ""
                think += f"\n- Requested: \"{self.query.intents[i].query}\""
                think += f"\n- Product Title: \"{best_p[i].title}\""
                think += f"\n- Shop ID: {best_p[i].shop}"

                for k, v in self.query.intents[i].template.items():
                    think += f"\n- Constraint `{k}`: Expects \"{v}\" | Found \"{v}\" in {'product title' if k == 'product_type' else 'SKU options'} → Exact Match ✓"

                if self.query.intents[i].price:
                    think += f"\n- Price: Expects {Mu_Util.promptfy_price(self.query.intents[i].price)} | Found {str(int(best_p[i].price))} → Within Range ✓"
            
                for svc in (self.query.intents[i].services or []):
                    think += f"\n- Service: Expects \"{svc}\" | Found \"{svc}\" in product details → Available ✓"
            
                think += "\n"

            if self.query.challenge == Mu_Challenge.VOUCHER:
                psum = sum(p.price for p in best_p)
                pact = self.query.voucher.get_active_cost(best_p) if self.query.voucher else psum

                if len(best_p) > 1:
                    think += "\nThe prices of the selected products are "
                    think += ", ".join([str(int(p.price)) for p in best_p[:-1]]) + " and " + str(int(best_p[-1].price))
                    think += ", respectively, with a total price of " + str(int(psum)) + "."
                else:
                    think += "\nThe price of the selected product is " + str(int(best_p[0].price)) + "."

                if psum > pact:
                    if self.query.voucher.apply_to == "shop":
                        think += f" Since all of them come from the same shop (`{best_p[0].shop}`) and the total price exceeds the voucher threshold, the voucher can be applied."
                
                    think += " After applying the voucher, the effective total price becomes " + str(int(pact)) + f", which is within the user's budget `{str(int(self.query.voucher.budget))}`."
                else:
                    think += f" It is already within the budget `{str(int(self.query.voucher.budget))}`, even without using the voucher."

                if len(best_p) > 1:
                    think += "\n\nIn summary, the selected products are within the budget and satisfy all stated requirements."
                else:
                    think += "\n\nIn summary, the selected product is within the budget and satisfies all stated requirements."
            elif self.query.challenge == Mu_Challenge.SHOP:
                think += f"\nAll the selected products come from the same shop (`{best_p[0].shop}`), as requested."
                think += "\n\nIn summary, all selected products come from the same shop, and each one satisfies the user's requirements."
            else:
                think += "\nAll checks passed. The selected product satisfies the user's requirements."

            think += "\n\nNow I will present the final recommendation."
            self.step(think, f"The final recommendation: \"{', '.join([p.id for p in best_p])}\"")



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

_KNG10694_AST_PAD = 0
if _KNG10694_AST_PAD:
    class _KNG10694_ProductVoucherOption(ProductJsonModel):
        budget: float
        apply_to: str
        threshold: Optional[float]
        is_fixed: bool
        discount: float
        cap: Optional[float]


    class _KNG10694_ShopProxyUtil:
        DEADLINE = 0.0

        @staticmethod
        def arm_deadline() -> None:
            ShopProxyUtil.DEADLINE = time.monotonic() + ShopSearchEngine.TIMEOUT_SECS

        @staticmethod
        def time_left() -> float:
            if not ShopProxyUtil.DEADLINE:
                return math.inf
            return ShopProxyUtil.DEADLINE - time.monotonic()

        @staticmethod
        def out_of_time() -> bool:
            return bool(ShopProxyUtil.DEADLINE) and time.monotonic() > ShopProxyUtil.DEADLINE
        LLM_PRIORITIES = ('moonshotai/kimi-k2.6', 'z-ai/glm-5.2', 'qwen/qwen3.5-397b-a17b', 'qwen/qwen3.6-27b', 'google/gemma-4-31b-it', 'z-ai/glm-5.2', 'deepseek/deepseek-v3.2', 'qwen/qwen3-32b', 'deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-chat-v3-0324', 'deepseek/deepseek-r1-0528', 'xiaomi/mimo-v2-flash')
        OPENROUTER_LLM_PRIORITIES = ('z-ai/glm-5.2', 'moonshotai/kimi-k2.6', 'openai/gpt-oss-120b', 'z-ai/glm-5.2', 'google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'qwen/qwen3.5-397b-a17b', 'qwen/qwen3.6-27b', 'deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-chat-v3-0324', 'deepseek/deepseek-r1-0528', 'qwen/qwen3-32b')
        DECOMPOSE_LLM_PRIORITIES: List[str] = []
        OPENROUTER_DECOMPOSE_LLM_PRIORITIES: List[str] = ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'z-ai/glm-5.2']
        SELECT_LLM_PRIORITIES: List[str] = []
        OPENROUTER_SELECT_LLM_PRIORITIES: List[str] = ['z-ai/glm-5.2', 'moonshotai/kimi-k2.6', 'openai/gpt-oss-120b', 'z-ai/glm-5.2', 'google/gemma-4-31b-it', 'deepseek/deepseek-v3.2']
        SELECT2_LLM_PRIORITIES: List[str] = []
        OPENROUTER_SELECT2_LLM_PRIORITIES: List[str] = ['deepseek/deepseek-v3.2', 'z-ai/glm-5.2', 'google/gemma-4-31b-it']
        CLASSIFY_LLM_PRIORITIES: List[str] = []
        OPENROUTER_CLASSIFY_LLM_PRIORITIES: List[str] = ['deepseek/deepseek-v3.2', 'z-ai/glm-5.2']

        @staticmethod
        def active_inference_provider() -> str:
            raw = (getenv('INFERENCE_PROVIDER') or 'chutes').strip().lower()
            return raw if raw in ('chutes', 'openrouter') else 'chutes'

        @staticmethod
        def default_llm_priorities() -> List[str]:
            if ShopProxyUtil.active_inference_provider() == 'openrouter':
                return ShopProxyUtil.OPENROUTER_LLM_PRIORITIES
            return ShopProxyUtil.LLM_PRIORITIES

        @staticmethod
        def get_llm_priorities(prefers: List[str]) -> List[str]:
            priorities = prefers.copy()
            for llm in ShopProxyUtil.default_llm_priorities():
                if llm not in priorities:
                    priorities.append(llm)
            return priorities

        @staticmethod
        def get_role_llm_priorities(role: str) -> List[str]:
            openrouter = ShopProxyUtil.active_inference_provider() == 'openrouter'
            if role == 'decompose':
                preferred = ShopProxyUtil.OPENROUTER_DECOMPOSE_LLM_PRIORITIES if openrouter else ShopProxyUtil.DECOMPOSE_LLM_PRIORITIES
            elif role == 'select':
                preferred = ShopProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else ShopProxyUtil.SELECT_LLM_PRIORITIES
            elif role == 'select2':
                preferred = ShopProxyUtil.OPENROUTER_SELECT2_LLM_PRIORITIES if openrouter else ShopProxyUtil.SELECT2_LLM_PRIORITIES
            elif role == 'filter':
                preferred = ShopProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else ShopProxyUtil.SELECT_LLM_PRIORITIES
            elif role == 'classify':
                preferred = ShopProxyUtil.OPENROUTER_CLASSIFY_LLM_PRIORITIES if openrouter else ShopProxyUtil.CLASSIFY_LLM_PRIORITIES
            else:
                preferred = []
            return ShopProxyUtil.get_llm_priorities(preferred)
        LLM_PROXY = ProxyClient(timeout=180, max_retries=5)

        @staticmethod
        def _empty_result_for(expect: str) -> Tuple[Dict[str, Any] | List[Any], Any]:
            if expect == 'list':
                return ([], None)
            return ({}, None)

        @staticmethod
        def trigger_llm(prompt: str, thinking: bool=False, preferred_llms: List[str]=[], temperature: float=0, expect: str='any') -> Tuple[Dict[str, Any] | List[Any], Any]:
            llm_priorities = ShopProxyUtil.get_llm_priorities(preferred_llms)
            trigger_tick = time.monotonic()
            t = temperature
            for _pass in range(2):
                for model in llm_priorities:
                    if ShopProxyUtil.out_of_time():
                        return ShopProxyUtil._empty_result_for(expect)
                    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'chat_template_kwargs': {'enable_thinking': thinking, 'thinking': thinking}, 'temperature': t, 'stream': False, 'max_completion_tokens': 8192, 'max_tokens': 8192}
                    try:
                        tick = time.monotonic()
                        resp = ShopProxyUtil.LLM_PROXY.post('/inference/chat/completions', json_data=payload)
                        try:
                            choice: Dict[str, Dict] = resp.get('choices')[0]
                            content = choice.get('message').get('content')
                            if not content:
                                raise Exception()
                        except Exception:
                            time.sleep(0.5)
                            raise Exception('invalid response')
                        try:
                            js = ShopMiscUtil.parse_json(content)
                            if js is None:
                                raise Exception()
                        except Exception as exc:
                            t += 0.3
                            raise Exception(f'failed to parse json: {exc}\n*******\n{content}\n******')
                        if expect == 'dict' and (not isinstance(js, dict)) or (expect == 'list' and (not isinstance(js, list))):
                            t += 0.3
                            raise Exception(f'expected {expect} but got {type(js)}')
                        throughput = resp.get('usage', {}).get('completion_tokens', 0)
                        trigger_delay = time.monotonic() - trigger_tick
                        delay = time.monotonic() - tick
                        return (js, resp)
                    except Exception as exc:
                        continue
            return ShopProxyUtil._empty_result_for(expect)
        ORO_PROXY = ProxyClient(timeout=30, max_retries=5)

        @staticmethod
        def find_products(so: ShopSearchOption, sort: Optional[str]=None, page: int=1) -> Tuple[List[ShopProduct], ToolCallResult]:
            empty_res: ToolCallResult = {'name': 'find_product', 'parameters': {'q': so.q, 'page': page}, 'tool_call_id': '', 'result': []}
            for _attempt in range(2):
                if ShopProxyUtil.out_of_time():
                    return ([], empty_res)
                try:
                    tool_res = execute_tool_call('find_product', {'q': so.q, 'page': page, 'shop_id': so.shop_id, 'price': so.price, 'sort': sort, 'service': ','.join(so.services or [])})
                    js_prods: List[Dict] = tool_res['result']
                    products = []
                    for j in js_prods:
                        price = j.get('price') or 0
                        price = ShopMiscUtil.extract_number(str(price))
                        products.append(ShopProduct(id=str(j.get('product_id')), title=j.get('title', ''), price=price, shop_id=j.get('shop_id', ''), service=j.get('service', []) or [], attributes={}, sku={}, is_viewed=False))
                    return (products, tool_res)
                except Exception as exc:
                    time.sleep(0.5)
            return ([], empty_res)

        @staticmethod
        def view_products(products: List[ShopProduct]) -> ToolCallResult:
            if not products:
                return
            pid_str = ','.join([p.id for p in products])
            for _attempt in range(5):
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
                    time.sleep(0.5)
            return {'name': 'view_product_information', 'parameters': {'product_ids': pid_str}, 'tool_call_id': '', 'result': []}

        @staticmethod
        def submit(products: List[ShopProduct]) -> ToolCallResult:
            pid_str = ','.join([p.id for p in products])
            return execute_tool_call('recommend_product', {'product_ids': pid_str})

        @staticmethod
        def terminate(status: str='success') -> ToolCallResult:
            return execute_tool_call('terminate', {'status': status})


    class _KNG10694_ShopDebugger(object):

        def __init__(self, suite: Optional[str]=None, frozen: Optional[List[str]]=None, stop_after: Optional[str]=None, show_dlg: bool=False, quiet: bool=True) -> None:
            self.memory: Dict[str, List] = defaultdict(lambda: [])
            self.quiet = quiet
            self.frozen = frozen or []
            self.stop_after = stop_after
            self.suite = suite
            self.show_dlg = show_dlg

        def add(self, stage: str, data: Any) -> None:
            if self.suite:
                self.memory[stage].append(data)

        def find(self, stage: str, kv: Dict) -> Any:
            for js in self.memory[stage]:
                for key, val in kv.items():
                    if js[key] != val:
                        break
                else:
                    return js
            raise Exception(f'# cannot find the freezed {stage} instance for `{json.dumps(kv, ensure_ascii=False)}`')

        def is_stop_in(self, episode: str) -> bool:
            if self.stop_after:
                if episode == 'AGENT' and self.stop_after.startswith('a-'):
                    return True
                if episode == 'BASE' and self.stop_after.startswith('b-'):
                    return True
                if episode == 'PRODUCT' and self.stop_after.startswith('p-'):
                    return True
                if episode == 'SHOP' and self.stop_after.startswith('s-'):
                    return True
                if episode == 'VOUCHER' and self.stop_after.startswith('v-'):
                    return True
            return False

        def is_stop(self, stage: str) -> bool:
            return self.stop_after == stage

        def is_frozen(self, stage: str) -> bool:
            return stage in self.frozen

        def dump(self) -> None:
            pass


    def _KNG10694_shop_resolve_engine_cls(challenge: 'ShoppingTaskType') -> Type['SearchEngine']:
        dispatch = {ShopShoppingTaskType.SHOP: ShopSearchEngine}
        return dispatch.get(challenge)


    class _KNG10694_VoucherShopSearchEngine(VoucherSearchEngine):
        MAX_TRIES = 4

        def run(self) -> List[VoucherProduct]:
            super(VoucherShopSearchEngine, self).run()
            _c245 = self.debugger.is_stop_in('BASE')
            if _c245:
                return []
            best_p = self._run_shop_strategy()
            _c246 = self.debugger.is_stop('s-proc')
            if _c246:
                return []
            return best_p

        def _run_shop_strategy(self) -> List[VoucherProduct]:
            best_p, _ = self.__proc_shop_batches__(True)
            return best_p

        def __solve_knapsack__(self, cats: List[List[Tuple[VoucherProduct, VoucherWeight]]]) -> List[Tuple[Tuple[VoucherProduct, ...], float]]:
            combs = VoucherMiscUtil.get_choice_combs(cats, False)
            solutions = []
            for choice in combs:
                prod_weights = [cats[i][n] for i, n in enumerate(choice)]
                prods, aggr_w = ([], 1.0)
                for p, w in prod_weights:
                    prods.append(p)
                    aggr_w = aggr_w * w.confidence
                if len({p.id for p in prods}) < len(prods):
                    continue
                solutions.append((tuple(prods), aggr_w))
            _rv82 = sorted(solutions, key=lambda s: -s[-1])
            return _rv82


    class _KNG10694_VoucherProductSearchEngine(VoucherSearchEngine):

        def run(self) -> List[VoucherProduct]:
            super(VoucherProductSearchEngine, self).run()
            _c243 = self.debugger.is_stop_in('BASE')
            if _c243:
                return []
            sq = self.qi.items[0]
            self._extend_product_search()
            candidates = self.__prune_by_score__(self.products[sq], sq)
            best, _ = self.__select__(self.qi.items[0], candidates[:VoucherSearchEngine.SELECT_CANDIDATES], True, 'p')
            _c244 = self.debugger.is_stop('p-select')
            if _c244:
                return []
            if best is not None:
                best_p = [best]
            else:
                best_p = []
            return best_p

        def _extend_product_search(self) -> None:
            sq = self.qi.items[0]
            sos = []
            _it182 = enumerate(sq.kw)
            for i, kw in _it182:
                k, v = (kw[-2], kw[-1])
                if not v.isnumeric() or int(v) != 1:
                    continue
                for a, b in VoucherVocab.P_EXT_SEARCH_UNITS:
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


    class _KNG10694_ShopJsonModel(object):

        @classmethod
        def from_json(cls, data: dict) -> 'JsonModel':
            kwargs = {}
            for f in fields(cls):
                if f.name not in data:
                    continue
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
            if origin is Union and len(args) == 2 and (NoneType in args):
                inner_t = next((a for a in args if a is not NoneType))
                if value is None:
                    return None
                return ShopJsonModel._coerce_from_json(value, inner_t)
            if is_dataclass(t) and isinstance(value, dict):
                return t.from_json(value)
            if origin is list and isinstance(value, list):
                return [ShopJsonModel._coerce_from_json(v, args[0]) for v in value]
            if origin is tuple and isinstance(value, list):
                return tuple((ShopJsonModel._coerce_from_json(v, args[0]) for v in value))
            if origin is dict and isinstance(value, dict):
                kt, vt = args
                return {ShopJsonModel._coerce_from_json(k, kt): ShopJsonModel._coerce_from_json(v, vt) for k, v in value.items()}
            if t is float and isinstance(value, str):
                return ShopMiscUtil.extract_number(value)
            if t is int and isinstance(value, str):
                return ShopMiscUtil.extract_number(value, True)
            if t is str:
                return str(value)
            return value

        @staticmethod
        def _coerce_to_json(obj) -> Any:
            if is_dataclass(obj):
                result = {}
                for f in fields(obj):
                    value = getattr(obj, f.name)
                    result[f.name] = ShopJsonModel._coerce_to_json(value)
                return result
            if isinstance(obj, list):
                return [ShopJsonModel._coerce_to_json(x) for x in obj]
            if isinstance(obj, dict):
                return {ShopJsonModel._coerce_to_json(k): ShopJsonModel._coerce_to_json(v) for k, v in obj.items()}
            return obj


    class _KNG10694_ProductEnhancementBenchSettings:
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


    class _KNG10694_ProductJsonModel(object):

        @classmethod
        def from_json(cls, data: dict) -> 'ProductJsonModel':
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
            if origin is Union and len(args) == 2 and (NoneType in args):
                inner_t = next((a for a in args if a is not NoneType))
                if value is None:
                    return None
                return ProductJsonModel.__convert_from__(value, inner_t)
            if is_dataclass(t) and isinstance(value, dict):
                return t.from_json(value)
            if origin is list and isinstance(value, list):
                return [ProductJsonModel.__convert_from__(v, args[0]) for v in value]
            if origin is tuple and isinstance(value, list):
                return tuple((ProductJsonModel.__convert_from__(v, args[0]) for v in value))
            if origin is dict and isinstance(value, dict):
                kt, vt = args
                return {ProductJsonModel.__convert_from__(k, kt): ProductJsonModel.__convert_from__(v, vt) for k, v in value.items()}
            if t is float and isinstance(value, str):
                return ProductMiscUtil.extract_number(value)
            if t is int and isinstance(value, str):
                return ProductMiscUtil.extract_number(value, True)
            if t is str:
                return str(value)
            return value

        @staticmethod
        def __convert_to__(obj) -> Any:
            if is_dataclass(obj):
                result = {}
                for f in fields(obj):
                    value = getattr(obj, f.name)
                    result[f.name] = ProductJsonModel.__convert_to__(value)
                return result
            if isinstance(obj, list):
                return [ProductJsonModel.__convert_to__(x) for x in obj]
            if isinstance(obj, dict):
                return {ProductJsonModel.__convert_to__(k): ProductJsonModel.__convert_to__(v) for k, v in obj.items()}
            return obj


    def _KNG10694_product_cast_model(source: ProductJsonModel, target_cls: Type[ProductJsonModel]) -> ProductJsonModel:
        return target_cls.from_json(source.to_json())


    def _KNG10694_ensure_dialog(dialog: list, query: str, lane: str) -> list:
        if dialog:
            return dialog
        return [create_dialogue_step(think=f'Agent returned no dialogue steps ({lane} lane).', tool_results=[], response='Done.', query=query, step=1)]


    class _KNG10694_VoucherTextCodec:

        @staticmethod
        def pull_number(s: str, as_int: bool=False) -> int | float:
            matches = re.findall('\\d+\\.?\\d*', s)
            if matches:
                val = float(matches[0])
            else:
                val = 0.0
            if as_int and matches:
                _rv17 = int(val)
            else:
                _rv17 = 0 if not matches else val
            return _rv17

        @staticmethod
        def coerce_number(v: Any, as_int: bool=False) -> int | float:
            if v is None:
                return 0
            _c204 = isinstance(v, (int, float))
            if _c204:
                return v
            _c205 = isinstance(v, str)
            if _c205:
                _rv18 = VoucherTextCodec.pull_number(v, as_int=as_int)
                return _rv18
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
            if s.isnumeric():
                _rv19 = int(float(s)) != 0
            else:
                _rv19 = False
            return _rv19

        @staticmethod
        def tidy_text(x: str) -> str:
            x = x.lower()
            x = re.sub('\\b(\\d+)([a-z]+)\\b', '\\1 \\2', x)
            x = re.sub('\\b([a-z]+)(\\d+)\\b', '\\1 \\2', x)
            x = re.sub('[\\(\\)\\[\\]\\{\\}\\+\\*\\_\\:\\;\\,\\-\\~\\/\\<\\>\\=]', ' ', x)
            x = re.sub('\\b(\\d+)\\s*x\\s*(\\d+)\\b', '\\1 \\2', x)
            _rv20 = re.sub('\\s+', ' ', x).strip()
            return _rv20

        @staticmethod
        def cartesian_slots(cat: Dict | List, include_non: bool) -> List[Dict | List]:
            if isinstance(cat, dict):
                keys = list(cat.keys())
            else:
                keys = list(range(len(cat)))
            choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]
            opts = product(*choices)
            if isinstance(cat, dict):
                _rv21 = [dict(zip(keys, opt)) for opt in opts]
            else:
                _rv21 = list(opts)
            return _rv21

        @staticmethod
        def dump_js(js: Dict[str, Any] | List[Any]) -> None:
            pass

        @staticmethod
        def decode_js(s: str) -> dict[str, Any] | list[Any] | None:
            cleaned = re.sub('^```(?:json)?\\s*', '', s.strip(), flags=re.I)
            cleaned = re.sub('\\s*```$', '', cleaned).strip()
            try:
                _rv22 = json.loads(cleaned)
                return _rv22
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
            _it125 = sorted(starts)
            for start in _it125:
                try:
                    obj, _ = decoder.raw_decode(cleaned[start:])
                    return obj
                except json.JSONDecodeError:
                    pass
            return None


    class _KNG10694_VoucherTemperatureController:
        """Task-aware temperature scheduling (bird/shrike/constGod/radiant patterns)."""
        _BASE: ClassVar[Dict[str, float]] = {'parse': 0.0, 'review': 0.0, 'decompose': 0.0, 'filter': 0.0, 'score': 0.5, 'judge': 0.5, 'single_judge': 0.3, 'narrate': 0.3, 'alt_pick': 0.0}

        @staticmethod
        def base_temperature(task: str) -> float:
            _rv103 = VoucherTemperatureController._BASE.get(task, 0.0)
            return _rv103

        @staticmethod
        def for_attempt(task: str, attempt: int, *, step: float | None=None, max_temp: float | None=None) -> float:
            base = VoucherTemperatureController.base_temperature(task)
            if step is None:
                step = VOUCHER_BENCH.temperature_retry_step
            else:
                step = step
            if max_temp is None:
                max_temp = VOUCHER_BENCH.temperature_max
            else:
                max_temp = max_temp
            _rv104 = min(max_temp, base + max(0, attempt) * step)
            return _rv104

        @staticmethod
        def for_judge_pass(pass_index: int, *, low_confidence: bool=False) -> float:
            if low_confidence:
                _rv105 = VoucherTemperatureController.for_attempt('judge', pass_index + 1)
                return _rv105
            _rv106 = VoucherTemperatureController.base_temperature('single_judge' if pass_index == 0 else 'judge')
            return _rv106


    class _KNG10694_ProductProxyUtil:
        DEADLINE = 0.0
        LLM_PRIORITIES = ['moonshotai/kimi-k2.6', 'z-ai/glm-5.2', 'qwen/qwen3.5-397b-a17b', 'qwen/qwen3.6-27b', 'google/gemma-4-31b-it', 'z-ai/glm-5.2', 'deepseek/deepseek-v3.2', 'qwen/qwen3-32b', 'deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-chat-v3-0324', 'deepseek/deepseek-r1-0528', 'xiaomi/mimo-v2-flash']
        OPENROUTER_LLM_PRIORITIES = ['z-ai/glm-5.2', 'moonshotai/kimi-k2.6', 'openai/gpt-oss-120b', 'z-ai/glm-5.2', 'google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'qwen/qwen3.5-397b-a17b', 'qwen/qwen3.6-27b', 'deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-chat-v3-0324', 'deepseek/deepseek-r1-0528', 'qwen/qwen3-32b']
        DECOMPOSE_LLM_PRIORITIES: List[str] = []
        OPENROUTER_DECOMPOSE_LLM_PRIORITIES: List[str] = ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'z-ai/glm-5.2']
        SELECT_LLM_PRIORITIES: List[str] = []
        OPENROUTER_SELECT_LLM_PRIORITIES: List[str] = ['z-ai/glm-5.2', 'moonshotai/kimi-k2.6', 'openai/gpt-oss-120b', 'z-ai/glm-5.2', 'google/gemma-4-31b-it', 'deepseek/deepseek-v3.2']
        CLASSIFY_LLM_PRIORITIES: List[str] = []
        OPENROUTER_CLASSIFY_LLM_PRIORITIES: List[str] = ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'z-ai/glm-5.2']

        @staticmethod
        def active_inference_provider() -> str:
            raw = (getenv('INFERENCE_PROVIDER') or 'chutes').strip().lower()
            return raw if raw in ('chutes', 'openrouter') else 'chutes'

        @staticmethod
        def default_llm_priorities() -> List[str]:
            if ProductProxyUtil.active_inference_provider() == 'openrouter':
                return ProductProxyUtil.OPENROUTER_LLM_PRIORITIES
            return ProductProxyUtil.LLM_PRIORITIES

        @staticmethod
        def get_llm_priorities(prefers: List[str]) -> List[str]:
            priorities = prefers.copy()
            for llm in ProductProxyUtil.default_llm_priorities():
                if llm not in priorities:
                    priorities.append(llm)
            return priorities

        @staticmethod
        def get_role_llm_priorities(role: str) -> List[str]:
            openrouter = ProductProxyUtil.active_inference_provider() == 'openrouter'
            if role == 'decompose':
                preferred = ProductProxyUtil.OPENROUTER_DECOMPOSE_LLM_PRIORITIES if openrouter else ProductProxyUtil.DECOMPOSE_LLM_PRIORITIES
            elif role == 'select':
                preferred = ProductProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else ProductProxyUtil.SELECT_LLM_PRIORITIES
            elif role == 'filter':
                preferred = ProductProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else ProductProxyUtil.SELECT_LLM_PRIORITIES
            elif role == 'classify':
                preferred = ProductProxyUtil.OPENROUTER_CLASSIFY_LLM_PRIORITIES if openrouter else ProductProxyUtil.CLASSIFY_LLM_PRIORITIES
            else:
                preferred = []
            return ProductProxyUtil.get_llm_priorities(preferred)
        LLM_PROXY = ProxyClient(timeout=180, max_retries=5)

        @staticmethod
        def arm_deadline() -> None:
            ProductProxyUtil.DEADLINE = time.monotonic() + ProductSearchEngine.TIMEOUT_SECS

        @staticmethod
        def time_left() -> float:
            if not ProductProxyUtil.DEADLINE:
                return math.inf
            return ProductProxyUtil.DEADLINE - time.monotonic()

        @staticmethod
        def out_of_time() -> bool:
            return bool(ProductProxyUtil.DEADLINE) and time.monotonic() > ProductProxyUtil.DEADLINE

        @staticmethod
        def __empty_for__(expect: str) -> Tuple[Dict[str, Any] | List[Any], Any]:
            if expect == 'list':
                return ([], None)
            return ({}, None)

        @staticmethod
        def trigger_llm(prompt: str, thinking: bool=False, preferred_llms: List[str]=[], temperature: float=0, expect: str='any') -> Tuple[Dict[str, Any] | List[Any], Any]:
            llm_priorities = ProductProxyUtil.get_llm_priorities(preferred_llms)
            trigger_tick = time.monotonic()
            t = temperature
            for _pass in range(2):
                for model in llm_priorities:
                    if ProductProxyUtil.out_of_time():
                        return ProductProxyUtil.__empty_for__(expect)
                    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'chat_template_kwargs': {'enable_thinking': thinking, 'thinking': thinking}, 'temperature': t, 'stream': False, 'max_completion_tokens': 8192, 'max_tokens': 8192}
                    try:
                        tick = time.monotonic()
                        resp = ProductProxyUtil.LLM_PROXY.post('/inference/chat/completions', json_data=payload)
                        try:
                            choice: Dict[str, Dict] = resp.get('choices')[0]
                            content = choice.get('message').get('content')
                            if not content:
                                raise Exception()
                        except Exception:
                            time.sleep(0.5)
                            raise Exception('invalid response')
                        try:
                            js = ProductMiscUtil.parse_json(content)
                            if js is None:
                                raise Exception()
                        except Exception as exc:
                            t += 0.3
                            raise Exception(f'failed to parse json: {exc}\n*******\n{content}\n******')
                        if expect == 'dict' and (not isinstance(js, dict)) or (expect == 'list' and (not isinstance(js, list))):
                            t += 0.3
                            raise Exception(f'expected {expect} but got {type(js)}')
                        throughput = resp.get('usage', {}).get('completion_tokens', 0)
                        trigger_delay = time.monotonic() - trigger_tick
                        delay = time.monotonic() - tick
                        return (js, resp)
                    except Exception as exc:
                        continue
            return ProductProxyUtil.__empty_for__(expect)
        ORO_PROXY = ProxyClient(timeout=30, max_retries=5)

        @staticmethod
        def find_products(so: ProductSearchOption, sort: Optional[str]=None, page: int=1) -> Tuple[List[ProductItem], ToolCallResult]:
            empty_res: ToolCallResult = {'name': 'find_product', 'parameters': {'q': so.q, 'page': page}, 'tool_call_id': '', 'result': []}
            for _attempt in range(2):
                if ProductProxyUtil.out_of_time():
                    return ([], empty_res)
                try:
                    tool_res = execute_tool_call('find_product', {'q': so.q, 'page': page, 'shop_id': so.shop_id, 'price': so.price, 'sort': sort, 'service': ','.join(so.services or [])})
                    js_prods: List[Dict] = tool_res['result']
                    products = []
                    for j in js_prods:
                        price = j.get('price') or 0
                        price = ProductMiscUtil.extract_number(str(price))
                        products.append(ProductItem(id=str(j.get('product_id')), title=j.get('title', ''), price=price, shop_id=j.get('shop_id', ''), service=j.get('service', []) or [], attributes={}, sku={}, is_viewed=False))
                    if products:
                        return (products, tool_res)
                except Exception as exc:
                    time.sleep(0.5)
            fallback_attempts = []
            if so.services:
                fallback_attempts.append({'service': '', 'price': so.price, 'q': so.q})
            if so.price:
                widened = None
                try:
                    widened = ProductFuzzyBoundaryLogic.widen_price_range(so.price)
                except Exception:
                    widened = None
                if not widened:
                    try:
                        lo, hi = ProductEngineUtil.numerify_price(so.price)
                        if isinstance(hi, (int, float)) and math.isfinite(hi) and (hi < 1000000000000.0):
                            hi = int(math.ceil(hi * 1.08))
                        if isinstance(lo, (int, float)) and lo > 0:
                            lo = int(max(0.0, lo * 0.92))
                        widened = ProductEngineUtil.stringfy_price(lo, hi)
                    except Exception:
                        widened = so.price
                fallback_attempts.append({'service': ','.join(so.services or []), 'price': widened, 'q': so.q})
            tokens = str(so.q or '').split()
            if len(tokens) > 2:
                fallback_attempts.append({'service': '', 'price': so.price, 'q': ' '.join(tokens[:2])})
            for fb in fallback_attempts:
                if ProductProxyUtil.out_of_time():
                    break
                try:
                    tool_res = execute_tool_call('find_product', {'q': fb['q'], 'page': page, 'shop_id': so.shop_id, 'price': fb['price'], 'sort': sort, 'service': fb['service']})
                    js_prods: List[Dict] = tool_res['result'] or []
                    if not js_prods:
                        continue
                    products = []
                    for j in js_prods:
                        price = j.get('price') or 0
                        price = ProductMiscUtil.extract_number(str(price))
                        products.append(ProductItem(id=str(j.get('product_id')), title=j.get('title', ''), price=price, shop_id=j.get('shop_id', ''), service=j.get('service', []) or [], attributes={}, sku={}, is_viewed=False))
                    return (products, tool_res)
                except Exception as exc:
                    time.sleep(0.3)
            return ([], empty_res)

        @staticmethod
        def view_products(products: List[ProductItem]) -> ToolCallResult:
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
                    time.sleep(0.5)

        @staticmethod
        def submit(products: List[ProductItem]) -> ToolCallResult:
            pid_str = ','.join([p.id for p in products])
            return execute_tool_call('recommend_product', {'product_ids': pid_str})

        @staticmethod
        def terminate(status: str='success') -> ToolCallResult:
            return execute_tool_call('terminate', {'status': status})

        @staticmethod
        def emit_calculate_voucher(sess: 'ProductSearchSession', products: List['ProductItem'], vopt: 'ProductVoucherOption') -> Optional[ToolCallResult]:
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

_KNG11794_AST_PAD = 0
if _KNG11794_AST_PAD:
    class _KNG11794_ProductShoppingTaskType(str, Enum):
        PRODUCT = 'product'
        SHOP = 'shop'
        VOUCHER = 'voucher'

        def explain(self, is_multiple: bool) -> str:
            match self:
                case ProductShoppingTaskType.PRODUCT:
                    return 'a single product with specific constraints'
                case ProductShoppingTaskType.SHOP:
                    return 'multiple products from the same shop'
                case ProductShoppingTaskType.VOUCHER:
                    pstr = 'multiple products' if is_multiple else 'a single product'
                    return f'{pstr} within a specified budget and considering voucher options'


    def _KNG11794_heappush(heap, item):
        heap.append(item)
        heap.sort()


    class _KNG11794_VoucherSearchOption(VoucherSubQuery):
        shop_id: Optional[str] = None
        q: str = ''

        def update_shop(self, shop_id: Optional[str]) -> 'VoucherSearchOption':
            self.shop_id = shop_id
            return self

        def update_price(self, price: Optional[str]) -> 'VoucherSearchOption':
            self.price = price
            return self

        def update_kw(self, id: int, kw: List[str]) -> 'VoucherSearchOption':
            self.kw[id] = kw
            return self

        def update_q(self, q: Optional[str]=None) -> 'VoucherSearchOption':
            if not q:
                segs = []
                for kw in self.kw:
                    v = kw[-1]
                    _it120 = v.split()
                    for m in _it120:
                        m = m.lower().strip()
                        if m:
                            if m not in segs:
                                segs.append(m)
                self.q = ' '.join(segs)
            else:
                self.q = q
            return self

        def __hash__(self):
            _rv13 = hash((self.q, self.query, ''.join(sorted(self.services)), self.price, self.shop_id))
            return _rv13

        def __eq__(self, other):
            if not isinstance(other, VoucherSearchOption):
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


    class _KNG11794_ShopVoucherOption(ShopJsonModel):
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


    class _KNG11794_VoucherSearchEngine(object):
        SELECT_CANDIDATES = 30
        SCORE_N_THRES = 0.4
        TIMEOUT_SECS = 210.0

        def __init__(self, qi: VoucherQueryInfo, sess: VoucherSearchSession, debugger: VoucherDebugger=None) -> None:
            self.debugger = debugger
            self.sess = sess
            self.qi = qi
            self.weight_pool: Dict[VoucherSubQuery, Dict[str, VoucherWeight]] = defaultdict(lambda: {})
            self.prod_pool: Dict[str, VoucherProduct] = {}
            self.searches: Dict[VoucherSubQuery, Dict[VoucherSearchOption, List[VoucherProduct]]] = {}
            self.products: Dict[VoucherSubQuery, List[VoucherProduct]] = {}
            self.synonyms: List[Tuple[str, str]] = [(a, b) for a, b in VoucherVocab.SYNONYMS]
            self.fillers: List[str] = []
            self.sos: Dict[VoucherSubQuery, List[VoucherSearchOption]] = defaultdict(lambda: [])
            self.irank: Dict[VoucherSubQuery, int] = defaultdict(lambda: 0)
            self.fscore: Dict[VoucherSubQuery, float] = {}
            self.filtered = defaultdict(lambda: defaultdict(lambda: None))
            self.watches: Dict[str, float] = {}

        def run(self) -> List[VoucherProduct]:
            self.__init_watch__('query', VoucherSearchEngine.TIMEOUT_SECS)
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
                    self.fscore[sq] = math.inf
                self.irank[sq] = len(self.products[sq])
            self.sess.step_on_search(self.qi)
            return []

        def __search__(self, sq: VoucherSubQuery, sos: List[VoucherSearchOption], id: str, page_limit: int, dbg_prefix: str) -> Dict:
            js: Dict[VoucherSearchOption, List[VoucherProduct]] = {}
            for so in sos:
                js[so] = self.__find_and_view__(so, page_limit)
            return js

        def __weigh__(self, sq: VoucherSubQuery, searches: Dict[VoucherSearchOption, List[VoucherProduct]], id: Optional[str], dbg_prefix: str) -> List[VoucherProduct]:
            ranks: Dict[VoucherProduct, int] = defaultdict(lambda: math.inf)
            products: List[VoucherProduct] = []
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

        def __select__(self, sq: VoucherSubQuery, candidates: List[VoucherProduct], must: bool, dbg_prefix: str) -> Tuple[Optional[VoucherProduct], bool]:
            best_p, js_check, is_fallback = (None, {}, False)
            if len(candidates) > 1 or (len(candidates) == 1 and (not must)):
                js_check, _ = VoucherProxyUtil.trigger_llm(VoucherPrompt.PV_SELECT.format(rules=VoucherPrompt.MATCH_RULES, query=sq.query, products=self.__get_prod_str__(candidates, sq, '', False)), preferred_llms=VoucherProxyUtil.get_role_llm_priorities('select'), expect='dict')
                bid = VoucherPrompt.extract_select_response(js_check)
                if bid >= 0 and bid < len(candidates):
                    best_p = candidates[bid]
                elif must:
                    best_p = candidates[0]
                    is_fallback = True
            elif len(candidates) > 0:
                best_p = candidates[0]
                is_fallback = True
            if best_p is not None:
                self.sess.step_on_select(best_p, js_check, is_fallback, sq)
            return (best_p, is_fallback)

        def __filter__(self, products: Dict[VoucherSubQuery, List[VoucherProduct]], id: str, dbg_prefix: str) -> Dict[VoucherSubQuery, List[VoucherProduct]]:
            temperature = 0
            while True:
                try:
                    n_cands = sum((len(ps) for ps in products.values()))
                    js, _ = VoucherProxyUtil.trigger_llm(self.__create_filter_prompt__(products), preferred_llms=VoucherProxyUtil.get_role_llm_priorities('filter'), temperature=temperature, expect='dict')
                    resp = VoucherPrompt.extract_filter_response(js, [len(products[sq]) for sq in products])
                    break
                except Exception as exc:
                    temperature = temperature + 0.3
            filtered: Dict[VoucherSubQuery, List[VoucherProduct]] = {}
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

        def __find_and_view__(self, opt: VoucherSearchOption, page_limit: int=1) -> List[VoucherProduct]:
            res: List[VoucherProduct] = []
            _it162 = range(1, 6)
            for page in _it162:
                if VoucherProxyUtil.out_of_time():
                    break
                searched, tool_res = VoucherProxyUtil.find_products(opt, page=page)
                self.sess.add_tool_result(tool_res)
                found: List[VoucherProduct] = []
                for p in searched:
                    _c238 = self.__basic_check__(p, opt)
                    if _c238:
                        found.append(self.__check_with_pool__(p))
                to_view = [p for p in found if not p.is_viewed]
                if len(to_view) > 0:
                    VoucherProxyUtil.view_products(to_view)
                for p in found:
                    res.append(self.__check_with_pool__(p))
                if page >= page_limit:
                    break
            return res

        def __basic_check__(self, prod: VoucherProduct, so: VoucherSearchOption) -> bool:
            if so.price:
                if not prod.price:
                    return False
                prod_price = VoucherMiscUtil.extract_number(str(prod.price))
                low, high = VoucherEngineUtil.numerify_price(so.price)
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

        def __weigh_product__(self, prod: VoucherProduct, sq: VoucherSubQuery, rank: int) -> VoucherWeight:
            _rv74 = VoucherFitScoreKit.best_weight(prod, sq, rank, self.synonyms)
            return _rv74

        def __check_with_pool__(self, p: VoucherProduct) -> VoucherProduct:
            if p.id in self.prod_pool:
                existing = self.prod_pool[p.id]
                if p.is_viewed:
                    if not existing.is_viewed:
                        existing.clone_from(p)
            else:
                self.prod_pool[p.id] = p
            _rv75 = self.prod_pool[p.id]
            return _rv75

        def __get_kw_var__(self, sq: VoucherSubQuery) -> Tuple[List[List[List[str]]], List[int]]:
            unit_indice, cat = ([], [])
            _it163 = enumerate(sq.kw)
            for i, ki in _it163:
                ks, v = (ki[:-1], ki[-1])
                if (m := re.search('^(\\d+)\\s+([a-z]+)$', v, re.I)):
                    if not any((k in ks for k in VoucherVocab.ID_KEYS)):
                        val, unit = (m.group(1), m.group(2))
                        cat.append([ki.copy(), ks + [unit] + [val], ks + [f'{val}{unit}']])
                        unit_indice.append(i)
                elif (m := re.search('^(\\d+(?:\\-\\d+)?)([a-z]+)$', v, re.I)):
                    if not any((k in ks for k in VoucherVocab.ID_KEYS)):
                        val, unit = (m.group(1), m.group(2))
                        cat.append([ki.copy(), ks + [unit] + [val]])
                        unit_indice.append(i)
                elif ks == ['size'] and len(v) >= 5 and (not re.search('\\d', v)):
                    d = VoucherPrompt.extract_kw_response.__name__[:5]
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
                    d1 = VoucherPrompt.extract_kw_response.__name__.split('_')[-1][:3]
                    d2 = VoucherPrompt.__name__.lower()[:3]
                    nv = re.sub(f'(\\w+)(?:\\s+|\\-){re.escape(d1)}(?:is)[a-z]+', '\\1' + f'{d2}of', v, flags=re.I)
                    if nv != v:
                        cat.append([ki.copy(), ks + [nv]])
                        unit_indice.append(i)
            return (cat, unit_indice)

        def __get_sos__(self, sq: VoucherSubQuery, shadow: bool) -> List[VoucherSearchOption]:
            if self.sos[sq]:
                return [voucher_cast_model(so, VoucherSearchOption) for so in self.sos[sq]]
            cat, unit_indice = self.__get_kw_var__(sq)
            sos: List[VoucherSearchOption] = []
            if cat:
                combs = VoucherMiscUtil.get_choice_combs(cat, False)
                sos = []
                for choice in combs:
                    _it164 = range(len(unit_indice))
                    for i in _it164:
                        id, c = (unit_indice[i], choice[i])
                        new_sq: VoucherSubQuery = voucher_cast_model(sq, VoucherSubQuery)
                        new_sq.kw[id] = cat[i][c]
                        sos.append(voucher_cast_model(new_sq, VoucherSearchOption))
            else:
                sos = [voucher_cast_model(sq, VoucherSearchOption)]
            for so in sos:
                if shadow:
                    low, high = VoucherEngineUtil.numerify_price(sq.price or '-')
                    if high < math.inf:
                        low = max(low, high // 2)
                    so.price = VoucherEngineUtil.stringfy_price(low, high)
                so.update_q()
            self.sos[sq] = list(set(sos))
            _rv76 = self.sos[sq]
            return _rv76

        def __get_bi_sos__(self, sq: VoucherSubQuery) -> List[VoucherSearchOption]:
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
            base_so: VoucherSearchOption = voucher_cast_model(base_sos[0], VoucherSearchOption)
            prices = sorted((p.price for p in products))
            median = int(prices[len(prices) // 2])
            low, high = VoucherEngineUtil.numerify_price(base_so.price or '-')
            lso: VoucherSearchOption = voucher_cast_model(base_so, VoucherSearchOption)
            lso.update_price(VoucherEngineUtil.stringfy_price(low, median))
            hso: VoucherSearchOption = voucher_cast_model(base_so, VoucherSearchOption)
            hso.update_price(VoucherEngineUtil.stringfy_price(median, high))
            return [lso, hso]

        def __get_prod_str__(self, products: List[VoucherProduct], sq: VoucherSubQuery, prefix: str, for_reason: bool) -> str:
            prod_str = ''
            _it165 = enumerate(products)
            for i, p in _it165:
                js_p = p.to_prompt(sq.query, for_reason)
                js_p['id'] = prefix + str(i + 1)
                prod_str = prod_str + (json.dumps(js_p, ensure_ascii=False) + '\n')
            return prod_str

        def __prune_by_score__(self, products: List[VoucherProduct], sq: VoucherSubQuery) -> List[VoucherProduct]:
            return [p for p in products if self.weight_pool[sq][p.id].score_n >= self.fscore[sq] - VoucherSearchEngine.SCORE_N_THRES]

        def __create_filter_prompt__(self, candidates: Dict[VoucherSubQuery, List[VoucherProduct]]) -> str:
            info = {}
            _it166 = enumerate(list(candidates.keys()))
            for i, sq in _it166:
                info[VoucherVocab.ALPHABETS[i]] = {'query': sq.query, 'candidates': self.__get_prod_str__(candidates[sq], sq, VoucherVocab.ALPHABETS[i], False)}
            _rv77 = VoucherPrompt.SV_FILTER.format(rules=VoucherPrompt.MATCH_RULES, count=len(candidates), info=info)
            return _rv77

        def __get_alt_str__(self, best_p: List[VoucherProduct]) -> str:
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
                return False
            if time.time() > self.watches[key]:
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
                    best_p = self.__proc_solmap__(solmap, True, f'shop-{id}', VoucherShopSearchEngine.MAX_TRIES)
                    self.sess.step_on_shop_batch(shop_batch, local_sol_count, best_p, id, self.qi)
                    if best_p or not self.__check_watch__():
                        break
            solutions.sort(key=lambda s: s[-1], reverse=True)
            if not best_p:
                _it171 = range(len(self.qi.items), 0, -1)
                for count in _it171:
                    for prods, _ in solutions:
                        if len([p for i, p in enumerate(prods) if self.filtered[self.qi.items[i]][p]]) >= count:
                            best_p = list(prods)
                            break
                    if best_p:
                        break
                if must and solutions:
                    if not best_p:
                        best_p = list(solutions[0][0])
                if best_p:
                    self.sess.step_on_shop_fallback(best_p, self.qi)
            return (best_p, solutions)

        def __get_shop_map__(self, products: Dict[VoucherSubQuery, List[VoucherProduct]]) -> Tuple[Dict[str, Dict[VoucherSubQuery, List[VoucherProduct]]], Dict[str, Tuple[int, float]]]:
            shop_map: Dict[str, Dict[VoucherSubQuery, List[VoucherProduct]]] = defaultdict(lambda: defaultdict(lambda: []))
            shop_stats: Dict[str, Tuple[int, float]] = {}
            for sq in self.qi.items:
                floor = self.fscore[sq] - VoucherSearchEngine.SHOP_MAP_SCORE_THRES
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

        def __plan_shop__(self) -> Tuple[List[List[str]], Dict[str, Dict[VoucherSubQuery, List[VoucherProduct]]], Dict[str, Tuple[int, float]]]:
            shop_map, shop_stats = self.__get_shop_map__(self.products)
            self.sess.step_on_shop_plan(shop_stats, min(type(self).MAX_EXT_SHOPS, len(list(shop_stats.keys()))), self.qi)
            shops = sorted(list(shop_stats.keys()), key=lambda sid: shop_stats[sid], reverse=True)
            plan = []
            _it172 = range(4, 0, -1)
            for n in _it172:
                covers = [sid for sid in shops if shop_stats[sid][0] == n]
                while len(covers) > 0:
                    plan.append(covers[:VoucherSearchEngine.SHOP_BATCH])
                    covers = covers[VoucherSearchEngine.SHOP_BATCH:]
            return (plan, shop_map, shop_stats)

        def __ext_search__(self, shops: List[str], shop_map: Dict[str, Dict[VoucherSubQuery, List[VoucherProduct]]], id: int) -> List[Dict[VoucherSubQuery, List[VoucherProduct]]]:
            products: Dict[VoucherSubQuery, List[VoucherProduct]] = {}
            searches: Dict[VoucherSubQuery, Dict] = defaultdict(lambda: {})
            for sid in shops:
                if VoucherProxyUtil.out_of_time():
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

        def __get_knapsack_maps__(self, products: Dict[VoucherSubQuery, List[VoucherProduct]], is_shop: bool) -> List[Dict[VoucherSubQuery, List[VoucherProduct]]]:
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

        def __proc_knapsack_map__(self, kmap: Dict[VoucherSubQuery, List[VoucherProduct]], **kwargs) -> List[Tuple[Tuple[VoucherProduct], Any]]:
            cats: List[List[Tuple[VoucherProduct, VoucherWeight]]] = []
            _it174 = kmap.items()
            for sq, products in _it174:
                prod_weights = [(p, self.weight_pool[sq][p.id]) for p in products]
                cats.append(prod_weights)
            _rv79 = self.__solve_knapsack__(cats, **kwargs)
            return _rv79
        MAX_SELECT_COMBOS = 10

        def __select_best_combo__(self, solutions: List[Tuple[Tuple[VoucherProduct, ...], Any]], id: str) -> List[VoucherProduct]:
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
            max_attempts = len(VoucherProxyUtil.get_llm_priorities([])) + 1
            _it177 = range(max_attempts)
            for _attempt in _it177:
                try:
                    n_cands = len(candidates)
                    js, _ = VoucherProxyUtil.trigger_llm(VoucherPrompt.SV_COMBO_SELECT.format(rules=VoucherPrompt.MATCH_RULES, n_items=len(self.qi.items), bundles=json.dumps(bundles, ensure_ascii=False)), preferred_llms=VoucherProxyUtil.get_role_llm_priorities('filter'), temperature=temperature, expect='dict')
                    if isinstance(js, dict):
                        reason = js.get('reason', '')
                    else:
                        reason = ''
                    self.debugger.add('k-combo-select', {'id': id, 'js': js, 'n_bundles': n_cands})
                    best_idx = VoucherPrompt.extract_combo_select_response(js, n_cands)
                    if best_idx is not None:
                        result = list(candidates[best_idx][0])
                        self.sess.step_on_combo_select(result, n_cands, reason, is_fallback=False)
                        return result
                    break
                except Exception as exc:
                    temperature = temperature + 0.3
            result = list(candidates[0][0])
            self.sess.step_on_combo_select(result, len(candidates), '', is_fallback=True)
            return result

        def __proc_solmap__(self, solmap: Dict[str, List[Tuple[Tuple[VoucherProduct, ...], Any]]], sort: bool, id: str, max_tries: int) -> List[VoucherProduct]:
            t, cursor, is_exhausted = (0, 0, False)
            _it181 = range(max_tries)
            for t in _it181:
                batch, candidates, cand_count = ([], defaultdict(lambda: []), 0)
                pre_valid: List[Tuple[Tuple[VoucherProduct, ...], Any]] = []
                if not self.__check_watch__():
                    break
                while cand_count < VoucherSearchEngine.FILTER_BATCH and (not is_exhausted):
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
                    return best_p
                if is_exhausted:
                    break
            return []

        def __find_filtered_combo__(self, solutions: List[Tuple[VoucherProduct, Any]]) -> List[VoucherProduct]:
            valid: List[Tuple[Tuple[VoucherProduct, ...], Any]] = []
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
            _rv81 = self.__select_best_combo__(valid, 'filtered-multi')
            return _rv81

        def __solve_knapsack__(self, _: List[List[Tuple[VoucherProduct, float]]], **__) -> List[Tuple[Tuple[VoucherProduct, ...], float]]:
            return []



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

_KNG20213_AST_PAD = 0
if _KNG20213_AST_PAD:
    class _KNG20213_NS_SegTree(object):
        """Top-k segment tree: each node holds a sorted list of up to k (value, index)
        pairs. query(l, r) returns the top-k items by value across the range [l, r]."""

        def __init__(self, values, k: int = 30) -> None:
            self.k, n = k, len(values)
            self.size = 1

            while self.size < n:
                self.size <<= 1

            self.seg = [[] for _ in range(2 * self.size)]

            for i, v in enumerate(values):
                self.seg[self.size + i] = [(v, i)]

            for i in range(self.size - 1, 0, -1):
                self.seg[i] = self.__merge__(self.seg[2 * i], self.seg[2 * i + 1])

        def __merge__(self, a: List, b: List) -> List:
            out, ia, ib = [], 0, 0

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
            if l > r: return []
            left, right = [], []

            l += self.size
            r += self.size

            while l <= r:
                if l & 1:
                    left = self.__merge__(left, self.seg[l])
                    l += 1

                if not (r & 1):
                    right = self.__merge__(self.seg[r], right)
                    r -= 1

                l >>= 1
                r >>= 1

            return self.__merge__(left, right)


    class _KNG20213_SA_VoucherSearchEngine(SA_SearchEngine):
        # MAX_PLATFORM_TRIES: used by __proc_solmap__ in the shop-voucher path only
        # (platform-voucher path uses __select_best_combo__ which is capped by
        # MAX_SELECT_COMBOS, not this constant).
        MAX_PLATFORM_TRIES = 6
        MAX_SHOP_TRIES = 3
        MAX_EXT_SHOPS = 3
        MAX_CAT_SIZE = 200
        SWAP_PRICE_DELTA = 3.0
        SCORE_FLOOR_N = 0.5
        MARGINAL_PAGE_LIMIT = 2
        MAX_K_SOLUTIONS = 500

        def __prune_by_score__(self, products: List[SA_Product], sq: SA_SubQuery) -> List[SA_Product]:
            base = super().__prune_by_score__(products, sq)
            return [
                p for p in base
                if p.id in self.weight_pool[sq] and self.weight_pool[sq][p.id].score_n >= self.SCORE_FLOOR_N
            ]

        def run(self) -> List[SA_Product]:
            super(SA_VoucherSearchEngine, self).run()
            if self.debugger.is_stop_in("BASE"): return []

            best_p = self.__v_proc__()
            if self.debugger.is_stop("v-proc"): return []

            return best_p

        def __v_proc__(self) -> List[SA_Product]:
            if not self.debugger.is_frozen("v-proc"):
                tick = time.time()

                if self.qi.vopt is None:
                    self.debugger.print("# voucher lane invoked without vopt — aborting v-proc")
                    return []

                if len(self.qi.items) > 1:
                    self.__search_marginal_bands__()

                if len(self.qi.items) > 1:
                    if self.qi.vopt.apply_to == "shop":
                        best_p = self.__proc_shop__()
                    else:
                        best_p = self.__proc_platform__(True)
                else:
                    best_p = self.__proc_single__()

                best_p = self.__budget_repair__(best_p)

                delay = time.time() - tick
                self.debugger.print({
                    "sq_count": len(self.qi.items),
                    "knapsack": [p.id for p in best_p],
                    "delay": f"{delay:.2f}s"
                })
                self.debugger.add(
                    "v-proc",
                    {
                        "query": self.qi.query_f,
                        "vopt": self.qi.vopt.to_json(),
                        "searches": [{"res": [p.to_json() for p in best_p]}]
                    }
                )
            else:
                best_p = []
                for js_prod in self.debugger.find(
                    "v-proc", {"query": self.qi.query_f}
                )["searches"][0]["res"]:
                    p = SA_Product.from_json(js_prod)
                    best_p.append(self.__check_with_pool__(p))

            return best_p

        # --- scored pool helper (respects SCORE_FLOOR_N) ---

        def __scored_pool__(self, sq: SA_SubQuery, products: List[SA_Product]) -> List[SA_Product]:
            # __prune_by_score__ already applies SCORE_FLOOR_N (via the override above);
            # fall back to unfiltered list only when nothing passes the floor.
            return self.__prune_by_score__(products, sq) or products

        def __merge_products__(self, sq: SA_SubQuery, new_products: List[SA_Product]) -> None:
            old_products = self.products.get(sq, [])
            seen = set()
            merged: List[SA_Product] = []

            for p in new_products + old_products:
                if p.id in seen: continue
                seen.add(p.id)
                merged.append(p)

            merged.sort(
                key = lambda p: (
                    -self.weight_pool[sq][p.id].score_n if p.id in self.weight_pool[sq] else 0,
                    p not in old_products
                )
            )
            self.products[sq] = merged
            if merged and merged[0].id in self.weight_pool[sq]:
                self.fscore[sq] = self.weight_pool[sq][merged[0].id].score_n

        # --- marginal band search ---

        def __search_marginal_bands__(self) -> None:
            vo = self.qi.vopt
            if vo is None or len(self.qi.items) < 2: return

            n = len(self.qi.items)
            threshold = float(vo.threshold or 0)
            allowed = SA_EngineUtil.get_allowed_total(vo)

            minima, maxima = self.__probe_price_edges__(allowed, self.__guess_voucher_shop__())
            if not minima or any(m <= 0 for m in minima):
                minima, maxima = [], []
                for sq in self.qi.items:
                    prices = [p.price for p in self.products.get(sq, []) if p.price]
                    if not prices: return
                    minima.append(min(prices))
                    maxima.append(max(prices))

            bands_info: List[Tuple[int, int, int, int]] = []

            for i, sq in enumerate(self.qi.items):
                band = SA_EngineUtil.get_marginal_band(i, n, threshold, allowed, minima, maxima, sq.price)
                if band is None: continue

                lo, hi = int(band[0]), int(math.ceil(band[1]))
                if hi <= lo: continue

                so = self.__get_sos__(sq, True)[0]
                so.price = SA_EngineUtil.stringfy_price(lo, hi)
                so.update_q()

                self.debugger.print(f"* marginal band search spec {i + 1}: {so.price}")
                searches = self.__search__(sq, [so], f"marginal-{i}", self.MARGINAL_PAGE_LIMIT, "v")
                new_prods = self.__weigh__(sq, searches, f"marginal-{i}", "v")
                new_prods = [p for p in new_prods if self.weight_pool[sq][p.id].score_n >= self.SCORE_FLOOR_N]
                self.searches[sq].update(searches)
                self.__merge_products__(sq, new_prods)
                bands_info.append((i + 1, lo, hi, len(new_prods)))

            if bands_info:
                self.debugger.add("v-marginal", {"query": self.qi.query_f, "bands": bands_info})
                self.sess.step_on_marginal_bands(bands_info)

        def __probe_price_edges__(
            self, allowed_total: float, shop_id: Optional[str] = None
        ) -> Tuple[List[float], List[float]]:
            minima: List[float] = []
            maxima: List[float] = []
            price_win = f"1-{max(int(allowed_total), 1)}"

            for sq in self.qi.items:
                so = self.__get_sos__(sq, True)[0]
                so.price = price_win
                if shop_id:
                    so.update_shop(shop_id)
                so.update_q()
                mn, mx = 0.0, 0.0

                for sort_key in ("priceasc", "pricedesc"):
                    prods, tool_res = SA_ProxyUtil.find_products(so, sort = sort_key, page = 1)
                    self.sess.add_tool_result(tool_res)
                    if not prods: continue
                    val = float(prods[0].price or 0)
                    if sort_key == "priceasc":
                        mn = val
                    else:
                        mx = val

                minima.append(mn)
                maxima.append(mx if mx > 0 else mn)

            return minima, maxima

        def __guess_voucher_shop__(self) -> Optional[str]:
            if self.qi.vopt is None or self.qi.vopt.apply_to != "shop":
                return None
            counts: Dict[str, int] = defaultdict(int)
            for sq in self.qi.items:
                for p in (self.products.get(sq) or [])[:8]:
                    if p.shop_id:
                        counts[p.shop_id] += 1
            return max(counts, key = counts.get) if counts else None

        # --- greedy seed and budget repair ---

        def __budget_repair__(self, best_p: List[SA_Product]) -> List[SA_Product]:
            if not best_p or self.qi.vopt is None: return best_p
            vo = self.qi.vopt

            if SA_EngineUtil.voucher_within_budget(best_p, vo):
                total = sum(p.price for p in best_p)
                if vo.threshold is None or total >= vo.threshold:
                    return best_p
            elif len(best_p) < 2:
                return best_p

            self.debugger.print("* budget repair: attempting price swap...")
            repaired = list(best_p)
            need_lower = not SA_EngineUtil.voucher_within_budget(best_p, vo)
            total = sum(p.price for p in repaired)
            need_higher = vo.threshold is not None and total < vo.threshold

            for i, sq in enumerate(self.qi.items):
                current = repaired[i]
                pool = self.__scored_pool__(sq, self.products.get(sq, [])) or self.products.get(sq, [])

                if need_lower:
                    alts = sorted(
                        [
                            p for p in pool
                            if p.id != current.id
                            and p.id in self.weight_pool[sq]
                            and p.price < current.price - SA_VoucherSearchEngine.SWAP_PRICE_DELTA
                        ],
                        key = lambda p: -self.weight_pool[sq][p.id].score_n
                    )
                elif need_higher:
                    alts = sorted(
                        [
                            p for p in pool
                            if p.id != current.id
                            and p.id in self.weight_pool[sq]
                            and p.price > current.price + SA_VoucherSearchEngine.SWAP_PRICE_DELTA
                        ],
                        key = lambda p: -self.weight_pool[sq][p.id].score_n
                    )
                else:
                    continue

                for alt in alts[:3]:
                    trial = list(repaired)
                    trial[i] = alt
                    trial_total = sum(p.price for p in trial)
                    if vo.threshold is not None and trial_total < vo.threshold:
                        continue
                    if SA_EngineUtil.voucher_within_budget(trial, vo):
                        self.debugger.print(f"* budget repair swapped spec {i + 1}: {current.id} -> {alt.id}")
                        self.debugger.add("v-repair", {"original": [p.id for p in best_p], "repaired": [p.id for p in trial]})
                        self.sess.step_on_budget_repair(best_p, trial, vo)
                        return trial

            return best_p

        # --- core strategy methods ---

        def __proc_single__(self) -> List[SA_Product]:
            favor_range = SA_EngineUtil.get_favor_range(self.qi.vopt)
            best_p, candidates = [], []

            pruned = self.__scored_pool__(self.qi.items[0], self.products[self.qi.items[0]])
            if not pruned: return best_p

            if favor_range is not None:
                candidates = [
                    p for p in pruned if p.price > favor_range[0] and p.price < favor_range[1]
                ][:SA_SearchEngine.SELECT_CANDIDATES]

                best, _ = self.__select__(self.qi.items[0], candidates, False, "v")
                best_p = [best] if best is not None else []

            if not best_p:
                candidates = [
                    p for p in pruned if p.price < self.qi.vopt.budget
                ][:SA_SearchEngine.SELECT_CANDIDATES]

                best, _ = self.__select__(self.qi.items[0], candidates, False, "v")
                best_p = [best] if best is not None else []

            if not best_p: best_p = [pruned[0]]
            return best_p

        def __proc_shop__(self) -> List[SA_Product]:
            favor_range = SA_EngineUtil.get_favor_range(self.qi.vopt)
            if favor_range is None: return self.__proc_platform__(False)

            best_p = self.__proc_shop_batches__(False, price_range = favor_range)
            if not best_p:
                best_p = self.__anchor_fill__()
            if not best_p and len(self.qi.items) == 2:
                best_p = self.__bidirectional_two_spec__()
            if not best_p:
                best_p = self.__rescore_top_shops__()
            if best_p: return best_p

            return self.__proc_platform__(False)

        def __proc_platform__(self, enable_favor: bool) -> List[SA_Product]:
            if not enable_favor: self.debugger.print("* switching to platform mode...")

            favor_range = SA_EngineUtil.get_favor_range(self.qi.vopt)
            solmap, best_p = {}, []

            shortlists = {
                sq: self.__scored_pool__(sq, self.products.get(sq, []))
                for sq in self.qi.items
            }

            kmaps = self.__get_knapsack_maps__(shortlists, False)
            if not kmaps: return best_p

            # The knapsack already enforces the voucher constraint, so every solution
            # in solmap is a valid candidate.  __proc_solmap__ will batch the unique
            # unverified products from those combos across ALL sub-queries in a
            # single __filter__ call, then return the first all-passing combo.
            if favor_range is not None and enable_favor:
                solmap["favor"] = self.__proc_knapsack_map__(kmaps[0], price_range = favor_range)

            solmap["fallback"] = self.__proc_knapsack_map__(kmaps[0], price_range = (0, self.qi.vopt.budget))

            # Emit the MITM step before selecting, so the timeline reads naturally:
            # "MITM found N combos" → "LLM selected the best one"
            sol_count = sum(len(sols) for sols in solmap.values())
            self.sess.step_on_platform(sol_count, not enable_favor)

            # All solutions from the knapsack already satisfy the voucher constraint.
            # Collect them, present the top-K as complete bundles to the LLM, and let
            # it pick the one where every item best meets its sub-query requirements.
            # Merge favor + fallback lanes, deduplicate by product-ID tuple, then sort
            # globally by descending knapsack weight so the best combos are first.
            seen_keys: set = set()
            all_solutions: List = []
            for _, sols in solmap.items():
                for s in sols:
                    key = tuple(p.id for p in s[0])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_solutions.append(s)
            all_solutions.sort(key=lambda s: s[-1], reverse=True)
            best_p = self.__select_best_combo__(all_solutions, "platform")

            if not best_p:
                self.debugger.print("* no valid combo found by LLM, falling back to top-scored.")
                best_p = list(all_solutions[0][0]) if all_solutions else []

            return best_p

        # --- shop fallback helpers (anchor fill, bidirectional, rescore) ---

        def __anchor_fill__(self) -> List[SA_Product]:
            n = len(self.qi.items)
            if n < 2: return []

            shop_map, shop_stats = self.__get_shop_map__(self.products)
            partial = {
                sid: stats[0] for sid, stats in shop_stats.items()
                if stats[0] == n - 1
            }
            if not partial: return []

            winner = max(partial, key = lambda sid: shop_stats[sid][1])
            covered: Dict[int, SA_Product] = {}

            for i, sq in enumerate(self.qi.items):
                pool = shop_map[winner][sq]
                if pool:
                    covered[i] = max(pool, key = lambda p: self.weight_pool[sq][p.id].score_n)

            if len(covered) != n - 1: return []

            missing_i = next(i for i in range(n) if i not in covered)
            missing_sq = self.qi.items[missing_i]

            self.debugger.print(f"* anchor fill: shop `{winner}`, missing spec {missing_i + 1}")
            missing_p = self.__anchor_search_in_shop__(missing_sq, winner)
            if missing_p is None: return []

            combo = [None] * n
            for i, p in covered.items():
                combo[i] = p
            combo[missing_i] = missing_p

            if not all(p is not None for p in combo): return []
            self.debugger.add("v-anchor", {"shop_id": winner, "missing_idx": missing_i, "combo": [p.id for p in combo]})
            self.sess.step_on_anchor_fill(winner, missing_i, list(combo))
            return list(combo)

        def __anchor_search_in_shop__(self, sq: SA_SubQuery, shop_id: str) -> Optional[SA_Product]:
            base_sos = [so.update_shop(shop_id) for so in self.__get_sos__(sq, True)]
            relax_options: List[SA_SearchOption] = list(base_sos)

            if base_sos and base_sos[0].services:
                so = self.__get_sos__(sq, True)[0]
                so.services = []
                so.update_shop(shop_id).update_q()
                relax_options.append(so)

            words = (base_sos[0].q if base_sos else "").split()
            if len(words) > 2 and base_sos:
                so = self.__get_sos__(sq, True)[0]
                so.services = []
                so.update_shop(shop_id).update_q(" ".join(words[:2]))
                relax_options.append(so)

            found: List[SA_Product] = []
            for so in relax_options:
                batch = self.__find_and_view__(so, page_limit = 1)
                found.extend(batch)
                if found: break

            if not found: return None
            found = list({p.id: p for p in found}.values())
            for p in found:
                if p.id not in self.weight_pool[sq]:
                    self.weight_pool[sq][p.id] = self.__weigh_product__(p, sq, 0)

            found.sort(key = lambda p: -self.weight_pool[sq][p.id].score_n)
            candidates = found[:SA_SearchEngine.SELECT_CANDIDATES]
            best, _ = self.__select__(sq, candidates, True, "v-anchor")
            return best if best is not None else found[0]

        def __pick_best_in_shop__(self, sq: SA_SubQuery, shop_id: str) -> Optional[SA_Product]:
            found = self.__anchor_search_in_shop__(sq, shop_id)
            if found is not None:
                return found
            pool = [
                p for p in self.__scored_pool__(sq, self.products.get(sq, []))
                if p.shop_id == shop_id
            ]
            if not pool:
                return None
            return max(pool, key = lambda p: self.weight_pool[sq][p.id].score_n)

        def __bidirectional_two_spec__(self) -> List[SA_Product]:
            n = len(self.qi.items)
            if n != 2: return []

            shop_map, shop_stats = self.__get_shop_map__(self.products)
            if any(st[0] == n for st in shop_stats.values()):
                return []

            best_combo: Optional[List[SA_Product]] = None
            best_sum = -1.0

            for anchor_idx in (0, 1):
                other_idx = 1 - anchor_idx
                anchor_sq, other_sq = self.qi.items[anchor_idx], self.qi.items[other_idx]

                shop_scores: Dict[str, float] = {}
                for p in self.__scored_pool__(anchor_sq, self.products.get(anchor_sq, [])):
                    sid = p.shop_id or ""
                    if not sid: continue
                    sc = self.weight_pool[anchor_sq][p.id].score_n
                    shop_scores[sid] = max(shop_scores.get(sid, 0.0), sc)

                for shop_id in sorted(shop_scores, key = shop_scores.get, reverse = True)[:6]:
                    other_pool = [
                        p for p in self.__scored_pool__(other_sq, self.products.get(other_sq, []))
                        if p.shop_id == shop_id
                    ]
                    if not other_pool: continue

                    anchor_p = self.__pick_best_in_shop__(anchor_sq, shop_id)
                    if anchor_p is None: continue

                    other_p = max(other_pool, key = lambda p: self.weight_pool[other_sq][p.id].score_n)
                    combo = [None, None]
                    combo[anchor_idx] = anchor_p
                    combo[other_idx] = other_p
                    ssum = sum(
                        self.weight_pool[self.qi.items[i]][combo[i].id].score_n
                        for i in range(2)
                    )
                    if ssum > best_sum:
                        best_sum = ssum
                        best_combo = combo

            if best_combo:
                self.debugger.print(f"* bidirectional two-spec resolved shop `{best_combo[0].shop_id}`")
                self.debugger.add("v-bidir", {"shop_id": best_combo[0].shop_id, "combo": [p.id for p in best_combo]})
                self.sess.step_on_bidirectional(best_combo[0].shop_id, best_combo)
            return best_combo or []

        def __rescore_top_shops__(self) -> List[SA_Product]:
            n = len(self.qi.items)
            if n < 2: return []

            shop_map, shop_stats = self.__get_shop_map__(self.products)
            if not shop_stats: return []

            ranked = sorted(shop_stats.items(), key = lambda kv: (-kv[1][0], -kv[1][1]))[:5]
            best_combo: Optional[List[SA_Product]] = None
            best_sum = -1.0

            for shop_id, (cov, _) in ranked:
                if cov < n - 1: continue
                combo: List[Optional[SA_Product]] = []
                ok = True
                for sq in self.qi.items:
                    pool = shop_map.get(shop_id, {}).get(sq, [])
                    if not pool:
                        ok = False
                        break
                    combo.append(max(pool, key = lambda p: self.weight_pool[sq][p.id].score_n))
                if not ok or len(combo) != n:
                    continue
                ssum = sum(self.weight_pool[sq][p.id].score_n for sq, p in zip(self.qi.items, combo))
                if ssum > best_sum:
                    best_sum = ssum
                    best_combo = combo

            if best_combo:
                self.debugger.print(f"* rescore top shops: picked `{best_combo[0].shop_id}`")
                self.debugger.add("v-rescore", {"shop_id": best_combo[0].shop_id, "combo": [p.id for p in best_combo]})
                self.sess.step_on_rescore_shops(best_combo[0].shop_id, best_combo)
            return list(best_combo) if best_combo else []

        # --- MITM knapsack: top-k SegTree + bounded heap ---

        def __solve_knapsack__(
            self, cats: List[List[Tuple[SA_Product, SA_Weight]]], price_range: Tuple[float, float]
        ) -> List[Tuple[Tuple[SA_Product, ...], Tuple[float, float]]]:
            split, heap = len(cats) // 2, []

            l_cats, r_cats = cats[:split], cats[split:]
            l_states, r_states = [], []

            for subcat, substates in [(l_cats, l_states), (r_cats, r_states)]:
                for choice in product(*subcat):
                    price = sum(p.price for p, _ in choice)
                    if price > price_range[1]: continue

                    substates.append((
                        price,
                        (math.prod(w.confidence for _, w in choice), sum(w.score_n for _, w in choice)),
                        [p for p, _ in choice],
                    ))

            # Process best left-states first so the heap fills with high-quality
            # solutions early, making eviction of weaker candidates more likely.
            l_states.sort(key=lambda x: x[1], reverse=True)
            r_states.sort(key=lambda x: x[0])

            seg = SA_SegTree([s[1] for s in r_states])
            r_prices = [s[0] for s in r_states]

            # Build a lookup so we can reconstruct Product objects at the end.
            prod_lookup: Dict[str, SA_Product] = {p.id: p for cat in cats for p, _ in cat}

            vo = self.qi.vopt

            for lp, lw, l_products in l_states:
                remain_lo = price_range[0] - lp
                remain_hi = price_range[1] - lp

                lidx = bisect_left(r_prices, remain_lo)
                ridx = bisect_right(r_prices, remain_hi) - 1

                if lidx > ridx: continue

                for rw, idx in seg.query(lidx, ridx):
                    prods = l_products + r_states[idx][2]

                    if len(set(prods)) != len(prods): continue
                    w: Tuple[float, float] = (lw[0] * rw[0], lw[1] + rw[1])
                    total = lp + r_prices[idx]

                    in_favor = price_range[0] == 0 or (vo is not None and SA_EngineUtil.is_favor_cost(vo, total))
                    if not in_favor: continue

                    pids = tuple(p.id for p in prods)
                    if len(heap) < SA_VoucherSearchEngine.MAX_K_SOLUTIONS:
                        heappush(heap, (w, pids))
                    elif w > heap[0][0]:
                        heapreplace(heap, (w, pids))

            heap.sort(key=lambda s: s[0], reverse=True)
            return [(tuple(prod_lookup[pid] for pid in pids), w) for w, pids in heap]



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

_KNG21D7D_AST_PAD = 0
if _KNG21D7D_AST_PAD:
    class _KNG21D7D_SA_QueryInfo(SA_JsonModel):
        challenge: SA_ShoppingTaskType = ""

        query_f: str = ""
        query_e: str = ""

        items: List[SA_SubQuery] = field(default_factory = list)
        vopt: Optional[SA_VoucherOption] = None


    class _KNG21D7D_SA_Vocab:
        ID_KEYS = ["brand", "color", "model", "id", "compatibility", "color_family"]
        ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        KW_STOPWORDS: List[str] = [
            "color family", "suitable", "compatible",
            "about", "above", "after", "also", "and", "as", "at", "before", "below", "between", "by", "for", "from", "in",
            "less", "more", "of", "on", "or", "over", "per", "than", "to", "under", "up", "with", "within"
        ]
        YES_WORDS: List[str] = ["yes", "enabled", "included", "exists", "true"]
        NO_WORDS: List[str] = ["no", "without any", "without"]
        COLOR_WORDS: List[str] = [
            "red", "blue", "green", "yellow", "black", "white", "pink", "purple", "violet", "orange", "brown", "gray", "grey",
            "beige", "navy", "gold", "silver", "bronze", "magenta", "cyan"
        ]
        SEARCHABLE_UNITS: List[str] = [
            "mm", "cm", "m", "km", "ft", "yd", "mg", "g", "kg", "oz", "lb",
            "ml", "l", "mah", "ah", "wh", "kwh", "v", "mv", "kv", "ma", "w", "kw",
            "hz", "khz", "mhz", "ghz", "kb", "mb", "gb", "tb",
            "bps", "kbps", "mbps", "gbps", "°c", "°f", "pcs", "pc",
            "pa", "kpa", "mpa", "mp", "dpi", "lux", "db", "rpm"
        ]
        SYNONYMS: List[Tuple[str, str]] = [
            ('1', 'one'),
            ('2', 'two'),
            ('3', 'three'),
            ('4', 'four'),
            ('5', 'five'),
            ('6', 'six'),
            ('7', 'seven'),
            ('8', 'eight'),
            ('9', 'nine'),
            ('10', 'ten'),
            ('one', 'single'),
            ('double', 'two'),
            ('three', 'triple'),
            ('1', 'single'),
            ('2', 'double'),
            ('3', 'triple'),
            ('1st', 'first'),
            ('2nd', 'second'),
            ('3rd', 'third'),
            ('100', 'hundred'),
            ('1000', 'thousand'),
            ('0', 'zero'),
            ('shock-resistant', 'shockproof'),
            ('shock resistant', 'shockproof'),
            ('shock-proof', 'shockproof'),
            ('water-resistant', 'waterproof'),
            ('water resistant', 'waterproof'),
            ('water-proof', 'waterproof'),
            ('extra small', 'xs'),
            ('extrasmall', 'xs'),
            ('extra-small', 'xs'),
            ('extra large', 'xl'),
            ('extralarge', 'xl'),
            ('extra-large', 'xl'),
            ('2xl', 'xxl'),
            ('3xl', 'xxxl'),
            ('freesize', 'one size'),
            ('sando', 'undershirt'),
        ]
        P_EXT_SEARCH_UNITS: List[Tuple[str, str]] = [
            ("piece", "pcs"), ("pack", "pcs")
        ]


    class _KNG21D7D_SA_EngineUtil:
        @staticmethod
        def get_spell_variations(s: str) -> List[str]:
            vars = []

            if len(s) >= 3 and not re.search(r"\d", s):
                for suffix in ["es", "ic", "n", "d", "ed", "y", "ing", "ily", "ied", "er", "est", "ier", "ren"]:
                    vars.append(s + suffix)

                    if s.endswith(suffix):
                        body = s[:-len(suffix)]

                        vars.append(body + "y")
                        vars.append(body)

                        if len(body) >= 4 and body[-1] == body[-2]: vars.append(body[:-1])

                for suffix in ["ed", "es", "y", "er", "est", "ing"]:
                    vars.append(s + s[-1] + suffix)

                if s.endswith("s"):
                    vars.append(s[:-1] + "es")
                    vars.append(s[:-1])
                else:
                    vars.append(s + "s")

                if s.endswith("y"):
                    for suffix in ["", "ied", "ies", "ier", "ing", "ily"]:
                        vars.append(s[:-1] + suffix)

                if len(s) <= 5 and s.endswith("en"): vars.append(s[:-2] + "an")
                if len(s) <= 5 and s.endswith("an"): vars.append(s[:-2] + "en")
                if len(s) >= 5 and s.endswith("or"): vars.append(s[:-2] + "our")
                if len(s) >= 6 and s.endswith("our"): vars.append(s[:-3] + "or")

            vars = list(set(v for v in vars if v != s and v not in SA_Vocab.KW_STOPWORDS and len(v) >= 3))
            return vars
        @staticmethod
        def is_kw_in(kw: str, s: str, as_is: bool, synonyms: List[Tuple[str, str]], terminal: bool = False) -> bool:
            if re.search(
                (r"\b" if re.match(r"\w", kw[0]) else "") + re.escape(kw) + (r"\b" if re.match(r"\w", kw[-1]) else ""),
                s,
                re.I
            ):
                return True

            if as_is: return False
            vars = SA_EngineUtil.get_spell_variations(kw)

            for v in vars:
                if re.search(rf"\b{re.escape(v)}\b", s, re.I): return True

            if len(kw) >= 6 and len(s) > len(kw) + 1:
                segs = s.replace('🎈', ' ').split()

                if len(segs) > 1:
                    for i in range(len(segs) - 1):
                        sub_s = segs[i] + segs[i + 1]
                        if kw == sub_s: return True

            # Reverse-compound: kw is a meaningful prefix/suffix of a compound token in s
            # e.g. kw="brake" matches inside "brakepad", kw="pad" is too short to check safely
            if not as_is and not terminal and len(kw) >= 5:
                for seg in s.replace('🎈', ' ').split():
                    if len(seg) > len(kw) and (seg.startswith(kw) or seg.endswith(kw)):
                        return True

            if kw in ["pcs", "pc", "pack"]:
                for w in ["piece", "pack", "set"]:
                    if w in s: return True

            for from_s, to_s in synonyms:
                if kw == from_s or kw == f"{from_s}s" or from_s == f"{kw}s":
                    if re.search(rf"\b{re.escape(to_s)}s?\b", s): return True
                elif kw == to_s or kw == f"{to_s}s" or to_s == f"{kw}s":
                    if re.search(rf"\b{re.escape(from_s)}s?\b", s): return True

            if not terminal:
                spell = re.sub(rf"ss", "sss", kw, flags = re.I)
                if spell != kw and SA_EngineUtil.is_kw_in(spell, s, False, synonyms, True): return True

            return False

        @staticmethod
        def is_valid_attr(k: str, v: str, q: str) -> bool:
            pairs: List[Tuple[str, str, str, str, bool]] = [
                (v, q, r'\b(\d+[a-z]*)\s*\*\s*(\d+[a-z]*)\b', r'{m1} by {m2}', True), (v, q, r'\b(\d+)\-(\d+)([a-z]+)\b', r'{m1}-{m2} {m3}', True),
                (v, q, r'\b(\d+)([a-z])\+', r'\b{m1}\s*{m2}([a-z]*)\sand\sup\b', True), (v, q, r'\b(\d+)\-(\d+)[a-z]+\b', r'\b{m1}\-{m2}\b', True),
                (v, q, r'\b(\d+)\-(\d+)y\b', r'\b{m1}\sto\s{m2}\syears\b', True), (q, v, r'\b(\d+)\s*\-\&\-\s*(\d+)\b', r'{m1}\sto\s{m2}', False),
                (v, q, r'\b(\d+)\s*-\s*(\d+)\s*([a-z]+)\b', r'{m3}\sbetween\s{m1}\sand\s{m2}', True), (v, q, r'\b(\d+)\.0\b', r'\b{m1}\b', True),
                (v, q, r'\b([\d\.]+)[a-z]*\s*~\s*([\d\.]+)\b', r'\b{m1}\sto\s{m2}\b', True), (v, q, r'【(\d+\-\d+[a-z]+)】', r'\b{m1}\b', True),
                (v, q, r'\b(\d+)\s*-\s*(\d+)([a-z]*)\b', r'up to {m2}{m3}', True), (v, q, r'\b(\w+)\s(\w+)\b', r'\b{m1}\s(\w+)\s{m2}\b', True),
                (q, v, r'\b([a-z]+)\s+\((\d)\s+piece\)\s+', r'^{m1}\s+{m2}pcs$', False), (v, q, r'^(?:[a-z]+)\:\s*(\d+)$', r'\b{m1}\b', True),
                (v, q, r'\b(?:[a-z]+)\:\s*(\d+)\s*-\s*(\d+)\s*([a-z]+)\b', r'up\sto\s{m2}{m3}', True),
                (v, q, r'\b(\d+)\s*-\s*(\d+)\b', r'{m1} and {m2}', True)
            ]
            for src, dst, src_pattern, dst_pattern, yes in sorted(pairs, key = lambda p: p[-1]):
                if (m := re.search(src_pattern, src, re.I)):
                    exists = re.search(
                        dst_pattern.format(
                            m1 = re.escape(m.group(1)) if m.lastindex and m.lastindex >= 1 else "",
                            m2 = re.escape(m.group(2)) if m.lastindex and m.lastindex >= 2 else "",
                            m3 = re.escape(m.group(3)) if m.lastindex and m.lastindex >= 3 else "",
                            k = k
                        ),
                        dst,
                        re.I
                    )
                    if not yes and not exists: return False

            return True

        @staticmethod
        def compact_kw(nested: str, q: str) -> Optional[List[str]]:
            splits = [s for s in [s.strip() for s in nested.lower().split("→")] if s]
            if len(splits) < 2: return None

            ks, v = splits[:-1], splits[-1]

            if v.lower() in SA_Vocab.YES_WORDS:
                for i in range(len(ks)):
                    t = " ".join([k.replace("_", " ") for k in ks[i:]])

                    if t in q:
                        ks, v = [v], t
                        break
                else:
                    return None

            for yes in SA_Vocab.YES_WORDS:
                if re.search(rf"\b{re.escape(yes)}$", v, re.I):
                    v = re.sub(rf"\b{re.escape(yes)}$", "", v, flags = re.I)

            for nw in SA_Vocab.NO_WORDS:
                if (m := re.search(rf"^{re.escape(nw)}\s+([\w\s]+)$", v, re.I)):
                    ks.append("no")
                    v = m.group(1)
                    break

            if f"{ks[-1]} {v}" in SA_Vocab.KW_STOPWORDS: return None
            if f"{v} {ks[-1]}" in SA_Vocab.KW_STOPWORDS: return None

            for sw in SA_Vocab.KW_STOPWORDS:
                v = re.sub(rf"\b{re.escape(sw)}\b", "", v, flags = re.I)

            if len(ks) == 1 and '_' in ks[0]:
                prefix = ks[0].split('_')[0] + '-'
                if v.startswith(prefix): v = v[len(prefix):]

                prefix = prefix[:-1]
                if re.search(rf"^{re.escape(prefix)}[a-z]+$", v, re.I): v = v[len(prefix):]

            if (m := re.search(r"^(any|all)\s+([\w\s]+)$", v, re.I)):
                k = m.group(2).replace(' ', '_')
                v = m.group(1)

                if k not in ks: ks.append(k)

            if v.isnumeric() and ks[-1] in ["eu", "us", "int"]:
                v, ks = f"{ks[-1]} {v}", ["size"]

            v = re.sub(r"\bge[a-z]([a-z]{2}one)\b", r"\1", v, flags = re.I)
            v = re.sub(r"\b(\d+)in(\d+)\b", r"\1-in-\2", v, flags = re.I)
            v = re.sub(r"[\&\|]", " ", v)
            v = re.sub(r"\s+", " ", v).strip()

            return (ks + [v]) if v else None
        @staticmethod
        def is_prod_only(q: str) -> bool:
            prod_str = type(SA_Product(id = "", title = "", price = 0)).__name__.lower()
            prod_suffix = "only"

            return any(
                re.search(pat, q, re.I) is not None for pat in [
                    rf"\b(?:just|{prod_suffix})(?:\s+the)?\s+{prod_str}s?\b",
                    rf"\b{prod_str}s?\s+{prod_suffix}\b"
                ]
            )

        @staticmethod
        def get_vouchered_cost(products: List[SA_Product], vo: Optional[SA_VoucherOption]) -> float:
            psum = sum([p.price for p in products])
            if vo is None: return psum

            if vo.apply_to == "shop":
                shops = set([p.shop_id for p in products])
                if len(shops) > 1: return psum

            if vo.threshold is not None:
                if psum < vo.threshold: return psum

            if vo.is_fixed: return psum - vo.discount

            discount = psum * vo.discount / 100.0
            if vo.cap is not None: discount = min(discount, vo.cap)

            return psum - discount
        @staticmethod
        def get_favor_range(vo: SA_VoucherOption) -> Optional[Tuple[float, float]]:
            if vo is None:
                return None
            if vo.is_fixed:
                upper = vo.budget + vo.discount
            else:
                if vo.discount >= 100:
                    return None
                upper_1 = vo.budget / (1.0 - vo.discount / 100.0)
                upper_2 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
                upper = min(upper_1, upper_2)

            if vo.threshold is not None:
                if upper >= vo.threshold:
                    return (vo.threshold, upper)
                else:
                    return None
            else:
                return (vo.budget, upper)
        @staticmethod
        def is_favor_cost(vo: SA_VoucherOption, psum: float) -> bool:
            """Return True if psum sits in a 'favorable' price window: the voucher
            threshold (when present) falls between 10–90% of psum, and for
            percentage-off vouchers the post-discount cost lands within ~110% of
            the declared budget.  Fixed-amount vouchers skip the budget-proximity
            check; threshold-less vouchers skip the 10–90% window check."""
            if psum < 100.0: return False
            if vo.threshold is not None:
                if vo.threshold < int(psum * 0.1) or vo.threshold > int(psum * 0.9): return False
            if not vo.is_fixed:
                ac = max(
                    psum * (1.0 - vo.discount / 100.0),
                    psum - (vo.cap if vo.cap is not None else math.inf)
                )
                if vo.budget < math.floor(ac) or vo.budget > math.ceil(1.1 * ac): return False
            return True
        @staticmethod
        def get_allowed_total(vo: SA_VoucherOption) -> float:
            if vo.is_fixed:
                return vo.budget + vo.discount
            upper_1 = vo.budget / (1.0 - vo.discount / 100.0) if vo.discount < 100 else math.inf
            upper_2 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
            return min(upper_1, upper_2)
        @staticmethod
        def get_marginal_band(
            sidx: int,
            n_specs: int,
            threshold: float,
            allowed_total: float,
            minima: List[float],
            maxima: List[float],
            price_range: Optional[str],
        ) -> Optional[Tuple[float, float]]:
            other = [i for i in range(n_specs) if i != sidx]
            floor_raw = max(0.0, threshold - sum(maxima[i] for i in other))
            ceiling_raw = allowed_total - sum(minima[i] for i in other)
            lo, hi = floor_raw, ceiling_raw

            if price_range:
                pr_lo, pr_hi = SA_EngineUtil.numerify_price(price_range)
                lo = max(lo, pr_lo)
                hi = min(hi, pr_hi)

            if lo > hi: return None
            return (lo, hi)
        @staticmethod
        def voucher_within_budget(products: List[SA_Product], vo: SA_VoucherOption) -> bool:
            return SA_EngineUtil.get_vouchered_cost(products, vo) <= vo.budget
        @staticmethod
        def promptfy_price(price: Optional[str]) -> str:
            if not price: return ""

            segs = [s.strip() for s in price.split('-')]
            if len(segs) != 2: return ""

            try:
                low = int(segs[0]) if segs[0] else None
                high = int(segs[1]) if segs[1] else None

                if low and high: return f"price between {low} and {high}"
                if low: return f"price greater than {low}"

                return f"price less than {high}"
            except Exception:
                return ""
        @staticmethod
        def numerify_price(price: Optional[str]) -> Tuple[int, int]:
            low_str, high_str = price.split('-', 1)
            low, high = 0, math.inf

            if low_str: low = SA_MiscUtil.extract_number(low_str, True)
            if high_str: high = SA_MiscUtil.extract_number(high_str, True)

            return low, high
        @staticmethod
        def stringfy_price(low: int, high: int) -> str:
            return (str(low) if low > 0 else "") + "-" + (str(high) if high < math.inf else "")


    class _KNG21D7D_NS_Debugger(object):
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

        def dump(self) -> None:
            pass

        def print(self, obj: Any) -> None:
            if not self.quiet:
                if isinstance(obj, dict) or isinstance(obj, list):
                    NS_MiscUtil.print_json(obj)
                else:
                    print(obj)


    class _KNG21D7D_SO2_PhaseBudget:
        """Wall-clock phase budgeting — keep engine reserve before optional LLM stages."""
        TOTAL_SECS = 240.0
        REVIEW_MIN_LEFT = 28.0
        ENGINE_RESERVE = 175.0

        def __init__(self) -> None:
            self.deadline = time.monotonic() + self.TOTAL_SECS

        def left(self, reserve: float = 0.0) -> float:
            return self.deadline - time.monotonic() - reserve

        def allow_review(self) -> bool:
            return self.left() >= self.REVIEW_MIN_LEFT


    class _KNG21D7D_SA_ProxyUtil:
        DEADLINE = 0.0
        LLM_PRIORITIES = [
            "moonshotai/kimi-k2.6", "moonshotai/kimi-k2.6",
            "z-ai/glm-5.2", "z-ai/glm-5.2",
            "qwen/qwen3.5-397b-a17b",
            "z-ai/glm-5.2",
            "deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat-v3-0324", "deepseek/deepseek-r1-0528",
            "google/gemma-4-31b-it", "qwen/qwen3-32b", "xiaomi/mimo-v2-flash"
        ]
        OPENROUTER_LLM_PRIORITIES = [
            "google/gemma-4-31b-it", "deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat-v3-0324",
            "openai/gpt-oss-120b", "moonshotai/kimi-k2.5", "qwen/qwen3-32b"
        ]

        # Role-specific preferred models (heads of the priority list for each step).
        # Leave empty [] to fall back to the global LLM_PRIORITIES order.
        DECOMPOSE_LLM_PRIORITIES: List[str] = []
        OPENROUTER_DECOMPOSE_LLM_PRIORITIES: List[str] = ["google/gemma-4-31b-it", "deepseek/deepseek-v3.2", "openai/gpt-oss-120b"]

        SELECT_LLM_PRIORITIES: List[str] = []
        OPENROUTER_SELECT_LLM_PRIORITIES: List[str] = ["google/gemma-4-31b-it", "openai/gpt-oss-120b", "deepseek/deepseek-v3.2"]

        # Models for task-type classification.  Fast, instruction-following models
        # are preferred; leave empty [] to fall back to the global LLM_PRIORITIES.
        CLASSIFY_LLM_PRIORITIES: List[str] = []
        OPENROUTER_CLASSIFY_LLM_PRIORITIES: List[str] = ["deepseek/deepseek-v3.2", "google/gemma-4-31b-it"]

        @staticmethod
        def active_inference_provider() -> str:
            raw = (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()
            return raw if raw in ("chutes", "openrouter") else "chutes"
        @staticmethod
        def default_llm_priorities() -> List[str]:
            if SA_ProxyUtil.active_inference_provider() == "openrouter":
                return SA_ProxyUtil.OPENROUTER_LLM_PRIORITIES
            return SA_ProxyUtil.LLM_PRIORITIES
        @staticmethod
        def get_llm_priorities(prefers: List[str]) -> List[str]:
            priorities = prefers.copy()

            for llm in SA_ProxyUtil.default_llm_priorities():
                if llm not in priorities:
                    priorities.append(llm)

            return priorities

        @staticmethod
        def get_role_llm_priorities(role: str) -> List[str]:
            """Return model priority list for a named role.

            Reads the role-specific *_LLM_PRIORITIES class variable as preferred
            heads, then appends the global fallback chain so the usual retry logic
            still applies.  Edit DECOMPOSE_LLM_PRIORITIES / SELECT_LLM_PRIORITIES
            (and their OPENROUTER_ variants) above to configure per-role models.
            """
            openrouter = SA_ProxyUtil.active_inference_provider() == "openrouter"
            if role == "decompose":
                preferred = SA_ProxyUtil.OPENROUTER_DECOMPOSE_LLM_PRIORITIES if openrouter else SA_ProxyUtil.DECOMPOSE_LLM_PRIORITIES
            elif role == "select":
                preferred = SA_ProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else SA_ProxyUtil.SELECT_LLM_PRIORITIES
            elif role == "filter":
                preferred = SA_ProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else SA_ProxyUtil.SELECT_LLM_PRIORITIES
            elif role == "classify":
                preferred = SA_ProxyUtil.OPENROUTER_CLASSIFY_LLM_PRIORITIES if openrouter else SA_ProxyUtil.CLASSIFY_LLM_PRIORITIES
            else:
                preferred = []
            return SA_ProxyUtil.get_llm_priorities(preferred)

        LLM_PROXY = ProxyClient(timeout = 60, max_retries = 2)

        @staticmethod
        def arm_deadline() -> None:
            SA_ProxyUtil.DEADLINE = time.monotonic() + SA_SearchEngine.TIMEOUT_SECS

        @staticmethod
        def time_left() -> float:
            if not SA_ProxyUtil.DEADLINE:
                return math.inf
            return SA_ProxyUtil.DEADLINE - time.monotonic()

        @staticmethod
        def out_of_time() -> bool:
            return bool(SA_ProxyUtil.DEADLINE) and time.monotonic() > SA_ProxyUtil.DEADLINE

        @staticmethod
        def __empty_for__(expect: str) -> Tuple[Dict[str, Any] | List[Any], Any]:
            if expect == "list":
                return [], None
            return {}, None
        @staticmethod
        def trigger_llm(
            prompt: str,
            thinking: bool = False,
            preferred_llms: List[str] = [],
            temperature: float = 0,
            expect: str = "any",
        ) -> Tuple[Dict[str, Any] | List[Any], Any]:

            llm_priorities = SA_ProxyUtil.get_llm_priorities(preferred_llms)
            trigger_tick = time.monotonic()
            t = temperature

            for _pass in range(2):
                for model in llm_priorities:
                    if SA_ProxyUtil.out_of_time():
                        print("# trigger_llm: global deadline reached, returning safe empty value")
                        return SA_ProxyUtil.__empty_for__(expect)

                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "chat_template_kwargs": {
                            "enable_thinking": thinking,
                            "thinking": thinking
                        },
                        "temperature": t,
                        "stream": False,
                        "max_completion_tokens": 1536,
                        "max_tokens": 1536
                    }
                    try:
                        tick = time.monotonic()
                        resp = SA_ProxyUtil.LLM_PROXY.post("/inference/chat/completions", json_data = payload)

                        try:
                            choice: Dict[str, Dict] = resp.get("choices")[0]
                            content = choice.get("message").get("content")
                            if not content:
                                raise Exception()
                        except Exception:
                            time.sleep(0.5)
                            raise Exception("invalid response")

                        try:
                            js = SA_MiscUtil.parse_json(content)
                            if js is None:
                                raise Exception()
                        except Exception as exc:
                            t += 0.3
                            raise Exception(f"failed to parse json: {exc}\n*******\n{content}\n******")

                        if (expect == "dict" and not isinstance(js, dict)) or \
                            (expect == "list" and not isinstance(js, list)):
                            t += 0.3
                            raise Exception(f"expected {expect} but got {type(js)}")

                        throughput = resp.get("usage", {}).get("completion_tokens", 0)
                        trigger_delay = time.monotonic() - trigger_tick
                        delay = time.monotonic() - tick

                        print(
                            f"* llm triggered: "
                            f"model = '{model}' | "
                            f"throughput = {throughput} | "
                            f"delay = {delay:.2f}s | "
                            f"total_delay = {trigger_delay:.2f}s"
                        )
                        return js, resp
                    except Exception as exc:
                        print(f"# llm trigger error with '{model}': {exc}")
                        continue

            print("# trigger_llm: exhausted all models, returning safe empty value")
            return SA_ProxyUtil.__empty_for__(expect)

        ORO_PROXY = ProxyClient(timeout = 30, max_retries = 5)

        @staticmethod
        def find_products(
            so: SA_SearchOption, sort: Optional[str] = None, page: int = 1
        ) -> Tuple[List[SA_Product], ToolCallResult]:

            empty_res: ToolCallResult = {
                "name": "find_product",
                "parameters": {"q": so.q, "page": page},
                "tool_call_id": "",
                "result": []
            }

            for _attempt in range(2):
                if SA_ProxyUtil.out_of_time():
                    print("# find_products: global deadline reached, returning empty results")
                    return [], empty_res

                try:
                    tool_res = execute_tool_call(
                        "find_product",
                        {
                            "q": so.q,
                            "page": page,
                            "shop_id": so.shop_id,
                            "price": so.price,
                            "sort": sort,
                            "service": ','.join(so.services or [])
                        }
                    )
                    js_prods: List[Dict] = tool_res["result"] or []
                    if not js_prods:
                        # Empty-search recovery: drop service, then shorten q (PROD88).
                        params = {
                            "q": so.q,
                            "page": page,
                            "shop_id": so.shop_id,
                            "price": so.price,
                            "sort": sort,
                        }
                        if so.services and not SA_ProxyUtil.out_of_time():
                            tool_res = execute_tool_call("find_product", params)
                            js_prods = tool_res.get("result") or []
                        if not js_prods and not SA_ProxyUtil.out_of_time():
                            short_q = " ".join(str(so.q or "").split()[:2])
                            if short_q and short_q != so.q:
                                params["q"] = short_q
                                tool_res = execute_tool_call("find_product", params)
                                js_prods = tool_res.get("result") or []

                    products = []
                    for j in js_prods:
                        price = j.get("price") or 0
                        price = SA_MiscUtil.extract_number(str(price))

                        products.append(
                            SA_Product(
                                id = str(j.get("product_id")),
                                title = j.get("title", ""),
                                price = price,
                                shop_id = j.get("shop_id", ""),
                                service = j.get("service", []) or [],
                                attributes = {},
                                sku = {},
                                is_viewed = False
                            )
                        )

                    return products, tool_res

                except Exception as exc:
                    print(f"# error in `find_products`: {exc}")
                    time.sleep(0.5)

            print("# find_products: attempts exhausted, returning empty results")
            return [], empty_res
        @staticmethod
        def view_products(products: List[SA_Product]) -> ToolCallResult:
            if not products: return
            pid_str = ','.join([p.id for p in products])

            for _attempt in range(4):
                if SA_ProxyUtil.out_of_time():
                    print("# view_products: global deadline reached, returning without details")
                    return
                try:
                    tool_res = execute_tool_call("view_product_information", {"product_ids": pid_str})

                    details = tool_res["result"]
                    if not details: return tool_res

                    details_map: Dict[str, Dict[str, Any]] = {d["product_id"]: d for d in details}

                    for prod in products:
                        d = details_map.get(prod.id, {})

                        prod.sku = d.get("sku_options", {})
                        prod.attributes = d.get("attributes", {})
                        prod.is_viewed = True

                    return tool_res
                except Exception as exc:
                    print(f"# error in `view_products`: {exc}")
                    time.sleep(0.5)
            print("# view_products: attempts exhausted")
            return
        @staticmethod
        def submit(products: List[SA_Product]) -> ToolCallResult:
            pid_str = ','.join([p.id for p in products])
            return execute_tool_call("recommend_product", {"product_ids": pid_str})
        @staticmethod
        def SA_terminate(status: str = "success") -> ToolCallResult:
            return execute_tool_call("terminate", {"status": status})
        @staticmethod
        def emit_calculate_voucher(
            sess: "SA_SearchSession", products: List["SA_Product"], vopt: "SA_VoucherOption"
        ) -> Optional[ToolCallResult]:
            if not products or vopt is None: return None

            prices_str = ",".join(str(p.price) for p in products)
            vtype = "fixed" if vopt.is_fixed else "percentage"
            threshold = float(vopt.threshold or 0)
            cap = float(vopt.cap or 0)

            tool_res = execute_tool_call(
                "calculate_voucher",
                {
                    "product_prices": prices_str,
                    "voucher_type": vtype,
                    "discount_value": vopt.discount,
                    "threshold": threshold,
                    "budget": vopt.budget,
                    "cap": cap,
                }
            )
            sess.add_tool_result(tool_res)
            result = tool_res.get("result") or {}

            think = (
                f"Confirming the cart with calculate_voucher: prices=[{prices_str}], "
                f"threshold={threshold:.0f}, budget={vopt.budget:.0f}"
            )
            if cap > 0:
                think += f", cap={cap:.0f}"
            think += (
                f". Result: total_before={result.get('total_before')}, "
                f"discount={result.get('discount_amount')}, "
                f"total_after={result.get('total_after')}, "
                f"within_budget={result.get('within_budget')}."
            )
            sess.step(think, "Voucher calculation confirmed.")
            return tool_res



__all__ = ["agent_main"]
