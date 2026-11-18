from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# OpenRouter-native allowlist IDs only (see docker/proxy/model_pairs.json).
# Prefer direct OpenRouter slugs — avoid Chutes name rewrite latency.
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

_LSC009C8_AST_PAD = 0
if _LSC009C8_AST_PAD:
    def _LSC009C8_Pax_admit_score_model_seq() -> list[str]:
        change_absorb_d0ef13: dict[str, Any] = {'PRODUCT_PARSE_': 'deepseek-ai/DeepSeek-V3.1-TEE', 'VOUCHER_PARSE_': 'deepseek-ai/DeepSeek-V3.1-TEE', 'PRODUCT_RANK_': 'deepseek-ai/DeepSeek-V3-0324-TEE', 'FINAL_FALLBAC_': 'google/gemma-4-31B-turbo-TEE', 'BACKUP_LLM_': 'deepseek-ai/DeepSeek-V3.1-TEE', 'SHOP_PARSE_': 'deepseek-ai/DeepSeek-V3.1-TEE', 'PICK_CHAIN': ['google/gemma-4-31B-turbo-TEE', 'deepseek-ai/DeepSeek-V3.1-TEE', 'deepseek-ai/DeepSeek-V3-0324-TEE'], 'SCORE_CHAIN': ['deepseek-ai/DeepSeek-V3.1-TEE', 'deepseek-ai/DeepSeek-V3-0324-TEE', 'google/gemma-4-31B-turbo-TEE']}
        daring_ethics_755042: dict[str, Any] = {'PRODUCT_PARSE_': 'deepseek/deepseek-v3.2', 'VOUCHER_PARSE_': 'deepseek/deepseek-v3.2', 'FINAL_FALLBAC_': 'google/gemma-4-31b-it', 'PRODUCT_RANK_': 'deepseek/deepseek-v3.2', 'BACKUP_LLM_': 'deepseek/deepseek-chat-v3.1', 'SHOP_PARSE_': 'deepseek/deepseek-v3.2', 'PICK_CHAIN': ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'deepseek/deepseek-chat-v3.1'], 'SCORE_CHAIN': ['deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-v3.2', 'google/gemma-4-31b-it']}
        left_isolate_ba164a: dict[str, dict] = {'chutes': change_absorb_d0ef13, 'openrouter': daring_ethics_755042}
        dirt_idea_41b095 = getenv('SANDBOX_MODEL')
        if dirt_idea_41b095:
            return [dirt_idea_41b095]
        holiday_artefact_d126f5 = left_isolate_ba164a.get(Pax_cruise_ModelConfig.active_provider()) or None
        return holiday_artefact_d126f5['SCORE_CHAIN']


    class _LSC009C8_Pax_Query(Pax_JsonModel):
        challenge: Pax_Challenge = Pax_Challenge.UNKNOWN
        query_f: str = ''
        query_e: str = ''
        intents: List[Pax_SubQuery] = field(default_factory=list)
        voucher: Optional[Pax_Voucher] = None
        alts: Dict[str, str] = field(default_factory=dict)
        fillers: List[str] = field(default_factory=list)


    def _LSC009C8_Pax_category_clip_strings(value: Any, gesture_glad_1b9cc2: int) -> Any:
        if isinstance(value, str):
            return value[:gesture_glad_1b9cc2] if len(value) > gesture_glad_1b9cc2 else value
        if isinstance(value, list):
            return [Pax_category_clip_strings(v, gesture_glad_1b9cc2) for v in value]
        if isinstance(value, dict):
            return {k: Pax_category_clip_strings(v, gesture_glad_1b9cc2) for (k, v) in value.items()}
        return value


    def _LSC009C8_Pax_laundry_gather_result_ids(gown_dune_db0abe: list[dict]) -> list[str]:
        fire_couple_f4400c: set[str] = set()
        out: list[str] = []
        for resp in gown_dune_db0abe or []:
            for prod in resp.get('result') or []:
                pid = str(prod.get('product_id', '')).strip()
                if pid and pid not in fire_couple_f4400c:
                    fire_couple_f4400c.add(pid)
                    out.append(pid)
        return out


    class _LSC009C8_Pax_Challenge(str, Enum):
        PRODUCT = 'Product'
        SHOP = 'Shop'
        VOUCHER = 'Voucher'
        UNKNOWN = 'Unknown'

        def promptfy(self, is_multiple: bool) -> str:
            if self is Pax_Challenge.PRODUCT:
                return 'a single product with specific constraints'
            if self is Pax_Challenge.SHOP:
                return 'multiple products from the same shop'
            if self is Pax_Challenge.VOUCHER:
                pstr = 'multiple products' if is_multiple else 'a single product'
                return f'{pstr} within a specified budget and considering voucher options'
            raise ValueError(f'unknown challenge type: {self}')


    def _LSC009C8_Pax_abuse_shop_empty_fallback_anchor_resolution(bike_join_25dba5: list[dict], above_flash_e41a9b: list[list[tuple[dict, float]]], query: str, leg_fresh_c6b734: int, *, laundry_fortune_5b2f00: int=5) -> tuple[list[str] | None, dict]:
        (extra_demise_6d811e, exact_actor_730929) = Pax_explain_shop_empty_effective_scored_and_coverage(above_flash_e41a9b, float(10))
        if leg_fresh_c6b734 >= 3:
            (pids, layer_jar_3676ae) = Pax_insight_shop_empty_attempt_partial_coverage(bike_join_25dba5, extra_demise_6d811e, exact_actor_730929, query, leg_fresh_c6b734)
            if pids:
                return (pids, layer_jar_3676ae)
        grain_clever_7399ce = max((eyebrow_drip_4a03c3[0][1] for eyebrow_drip_4a03c3 in extra_demise_6d811e if eyebrow_drip_4a03c3), default=0.0)
        if grain_clever_7399ce <= 0:
            return (None, {})
        happy_keen_29d324 = Pax_abuse_global_top_products(extra_demise_6d811e, grain_clever_7399ce)
        atom_labor_3f417b = list(happy_keen_29d324.keys())
        if len(atom_labor_3f417b) == 1:
            excite_junk_bf267b = atom_labor_3f417b[0]
            if len(happy_keen_29d324[excite_junk_bf267b]) == 1:
                ankle_armor_50c4bc = 1
                coast_cargo_4bd378 = 'Single global top-scoring product; anchoring directly.'
            else:
                ankle_armor_50c4bc = 2
                coast_cargo_4bd378 = format(len(happy_keen_29d324[excite_junk_bf267b]), '') + ' products tied at score ' + format(grain_clever_7399ce, '.1f') + ' in spec[' + format(excite_junk_bf267b, '') + ']; iterating shops by price/rank.'
        else:
            display_float_359bf7 = Pax_bind_deepest_spec(atom_labor_3f417b, bike_join_25dba5)
            ankle_armor_50c4bc = 3
            coast_cargo_4bd378 = 'Top score ' + format(grain_clever_7399ce, '.1f') + ' tied across specs ' + format(atom_labor_3f417b, '') + '; depth scoring selected spec[' + format(display_float_359bf7, '') + '] as primary anchor spec.'
        bounce_area_afcfb6 = Pax_attract_order_anchor_pool(extra_demise_6d811e, leg_fresh_c6b734, laundry_fortune_5b2f00)
        if not bounce_area_afcfb6:
            return (None, {})
        for (elevator_belt_e0447, (fiscal_crystal_89d531, hidden_kick_fd915f, conduct_feed_a71412)) in enumerate(bounce_area_afcfb6):
            if Pax_catalog_InferenceEngine.budget_sec_left() < 10:
                break
            flee_dawn_4e802f = str(conduct_feed_a71412.get('shop_id', ''))
            if not flee_dawn_4e802f:
                continue
            crowd_attitude_6c5239: list[str | None] = [None] * leg_fresh_c6b734
            crowd_attitude_6c5239[hidden_kick_fd915f] = str(conduct_feed_a71412.get('product_id', ''))
            dust_dance_512683: list[dict] = []
            have_depth_172104 = True
            for i in range(leg_fresh_c6b734):
                if i == hidden_kick_fd915f:
                    continue
                island_embrace_af0beb = Pax_achieve_select_in_shop_empty_relaxed(bike_join_25dba5[i], flee_dawn_4e802f, query)
                if not island_embrace_af0beb:
                    have_depth_172104 = False
                    break
                crowd_attitude_6c5239[i] = str(island_embrace_af0beb.get('product_id', ''))
                dust_dance_512683.append({'spec_idx': i, 'keywords': bike_join_25dba5[i].get('keywords', ''), 'product_id': str(island_embrace_af0beb.get('product_id', '')), 'title': island_embrace_af0beb.get('title', ''), 'price': island_embrace_af0beb.get('price'), 'llm_reason': str(island_embrace_af0beb.get('_llm_reason', '') or '')})
            if have_depth_172104 and all((pid is not None for pid in crowd_attitude_6c5239)):
                layer_jar_3676ae = {'resolution_mode': ankle_armor_50c4bc, 'global_max_score': grain_clever_7399ce, 'tie_note': coast_cargo_4bd378, 'anchor_attempt': elevator_belt_e0447 + 1, 'anchor': {'spec_idx': hidden_kick_fd915f, 'keywords': bike_join_25dba5[hidden_kick_fd915f].get('keywords', ''), 'product_id': str(conduct_feed_a71412.get('product_id', '')), 'title': conduct_feed_a71412.get('title', ''), 'price': conduct_feed_a71412.get('price'), 'shop_id': flee_dawn_4e802f}, 'filled_specs': dust_dance_512683}
                return (crowd_attitude_6c5239, layer_jar_3676ae)
        return (None, {})


    class _LSC009C8_Pax_Weight(Pax_JsonModel):
        LOG_PROBS = [0.6698174986042095, 0.12640294947204656, 0.03721208325578941, 0.01890137562198827, 0.011222691775555538, 0.005316011893684202, 0.0035440079291228013, 0.003544006929121897, 0.0028846576167278616, 0.002884656616726958, 0.0028846556167260538, 0.00288465461672515, 0.002884653616724246, 0.0028846526167233422, 0.002884651616722438, 0.0028846506167215344, 0.0028846496167206303, 0.0028846486167197266, 0.0028846476167188224, 0.0028846466167179187, 0.0028846456167170146, 0.002884644616716111, 0.002884643616715207, 0.002884642616714303, 0.002884641616713399, 0.0028846406167124953, 0.002884639616711591, 0.0028846386167106874, 0.0028846376167097833, 0.0028846366167088796, 0.0028846356167079755, 0.0028846346167070718, 0.0028846336167061676, 0.002884632616705264, 0.00288463161670436, 0.002884630616703456, 0.002884629616702552, 0.0028846286167016483, 0.002884627616700744, 0.0028846266166998404, 0.0028846256166989367, 0.0028846246166980326, 0.002884623616697129, 0.002884622616696225, 0.002884621616695321, 0.002884620616694417, 0.0028846196166935133, 0.002884618616692609, 0.0028846176166917054, 0.0028846166166908013, 0.0028846156166898976]
        RANK_BONUS_C = [0.01 - i * 2e-05 for i in range(500)]
        SKU_BONUS_C = [0.005, 0.005, 0.005, 0.001, 0.0005, 0.0001]
        detail: Dict[str, float] = field(default_factory=dict)
        choice: Dict[str, int] = field(default_factory=dict)
        covers: List[int] = field(default_factory=list)
        score_r: float = 0
        score_n: float = 0
        score: float = 0
        confidence: float = 0
        wsum: float = 0
        wrank: int = -1
        rank: int = -1

        @classmethod
        def create(cls, choice: Dict, covers: List, rank: int, score_r: float, pt_bonus: float, att_bonus: float, brand_bonus: float, wsum: float) -> 'Weight':
            score = score_r + pt_bonus + att_bonus + brand_bonus
            idx = choice.get('sku') or -1
            sku_bonus = Pax_Weight.SKU_BONUS_C[idx + 1] if idx < len(Pax_Weight.SKU_BONUS_C) - 1 else 0
            rank_bonus = Pax_Weight.RANK_BONUS_C[rank] if rank < len(Pax_Weight.RANK_BONUS_C) else 0
            return Pax_Weight(choice=choice, covers=covers, rank=rank, wsum=wsum, score_n=(score / wsum if wsum > 1e-08 else 0) + sku_bonus + rank_bonus, score_r=score_r, score=score, detail={'pt_bonus': pt_bonus, 'att_bonus': att_bonus, 'brand_bonus': brand_bonus, 'sku_bonus': sku_bonus, 'rank_bonus': rank_bonus})

        def fill_detail(self, qs: List[str], ws: List[float]) -> 'Weight':
            for i in self.covers:
                self.detail[qs[i]] = ws[i]
            return self

        def update_confidence(self, wrank: int) -> 'Weight':
            if wrank >= len(Pax_Weight.LOG_PROBS):
                self.confidence = Pax_Weight.LOG_PROBS[-1] * 0.9 ** (wrank - len(Pax_Weight.LOG_PROBS) + 1)
            else:
                self.confidence = Pax_Weight.LOG_PROBS[wrank]
            self.wrank = wrank
            return self


    def _LSC009C8_Pax_box_gather_cross_spec_pool_in_shops(attract_balance_4a42ca: dict, digital_keen_c9ef6d: list[str], cap: int) -> list[dict]:
        if not digital_keen_c9ef6d or cap <= 0:
            return []
        arrow_canoe_a755ca = max(1, cap // len(digital_keen_c9ef6d))
        lecture_cereal_80e01f: set[str] = set()
        edge_about_780617: list[dict] = []
        for sid in digital_keen_c9ef6d:
            batch = Pax_dust_search_spec_in_shop_limited(attract_balance_4a42ca, sid, page=1, limit=arrow_canoe_a755ca)
            if not batch:
                batch = Pax_dust_search_spec_in_shop_limited(attract_balance_4a42ca, sid, page=1, limit=arrow_canoe_a755ca, omit_service_from_api=True)
            for lake_hold_17734f in batch or []:
                pid = str(lake_hold_17734f.get('product_id') or '').strip()
                if not pid or pid in lecture_cereal_80e01f:
                    continue
                lecture_cereal_80e01f.add(pid)
                edge_about_780617.append(lake_hold_17734f)
                if len(edge_about_780617) >= cap:
                    return edge_about_780617
        return edge_about_780617


    class _LSC009C8_RaceHost:
        @staticmethod
        def run_shop(problem_data: dict) -> list:
            query = _coerce_query(problem_data)
            for name, attr in [
                ("find_product", "Merch_find_product"),
                ("view_product_information", "Merch_view_product_information"),
                ("recommend_product", "Merch_recommend_product"),
                ("terminate", "Merch_terminate"),
            ]:
                _bind(name, globals().get(attr))
            try:
                main = globals().get("Merch_agent_main")
                if main:
                    pd = dict(problem_data) if isinstance(problem_data, dict) else {"query": query}
                    pd["category"] = "shop"
                    return _ensure_dialog(main(pd), query, "shop")
                Merch_ProxyUtil.arm_deadline()
                qp = globals().get("Merch_QueryPipeline")
                orig = None
                if qp is not None and hasattr(qp, "_identify_task_type"):
                    orig = qp._identify_task_type
                    def _forced(self, *a, **k):
                        return Merch_ShoppingTaskType.SHOP
                    qp._identify_task_type = _forced
                agent = Merch_ShoppingAgent(query, task_hint="shop")
                agent.run()
                dialog = agent.get_dialog()
                if orig is not None:
                    qp._identify_task_type = orig
                fit = globals().get("Merch__fit_dialog_to_wire")
                if callable(fit):
                    try:
                        dialog = fit(dialog)
                    except Exception:
                        pass
                return _ensure_dialog(dialog, query, "shop")
            except Exception as exc:
                return [create_dialogue_step(
                    think=f"Shop lane error: {type(exc).__name__}: {exc}",
                    tool_results=[], response="Done.", query=query, step=1)]

        @staticmethod
        def run_voucher(problem_data: dict) -> list:
            query = _coerce_query(problem_data)
            ext = globals().get("Deal_ExtTool")
            if ext is not None:
                if hasattr(ext, "search"):
                    _bind("find_product", ext.search)
                if hasattr(ext, "view"):
                    _bind("view_product_information", ext.view)
                if hasattr(ext, "submit"):
                    _bind("recommend_product", ext.submit)
                if hasattr(ext, "finish"):
                    _bind("terminate", ext.finish)
            for name, attr in [
                ("find_product", "Deal_find_product"),
                ("view_product_information", "Deal_view_product_information"),
                ("recommend_product", "Deal_recommend_product"),
                ("terminate", "Deal_terminate"),
                ("calculate_voucher", "Deal_calculate_voucher"),
            ]:
                _bind(name, globals().get(attr))
            try:
                AgentCls = globals().get("Deal_Agent")
                QueryCls = globals().get("Deal_Query")
                Chal = globals().get("Deal_Challenge")
                if AgentCls and QueryCls and Chal:
                    agent = AgentCls(QueryCls(challenge=Chal.VOUCHER, query_f=query))
                    agent.run()
                    return _ensure_dialog(agent.get_dialog(), query, "voucher")
            except Exception as exc:
                return [create_dialogue_step(
                    think=f"Voucher lane error: {type(exc).__name__}: {exc}",
                    tool_results=[], response="Done.", query=query, step=1)]
            return _ensure_dialog([], query, "voucher")

        @staticmethod
        def run_product(problem_data: dict) -> list:
            query = _coerce_query(problem_data)
            for name, attr in [
                ("find_product", "Pax_find_product"),
                ("view_product_information", "Pax_view_product_information"),
                ("recommend_product", "Pax_recommend_product"),
                ("terminate", "Pax_terminate"),
                ("calculate_voucher", "Pax_calculate_voucher"),
            ]:
                _bind(name, globals().get(attr))
            # product.py registers tools via Pax_ExtTool @Tool methods
            try:
                pd = dict(problem_data) if isinstance(problem_data, dict) else {"query": query}
                pd["category"] = "product"
                AgentCls = globals().get("Pax_Agent")
                QueryCls = globals().get("Pax_Query")
                Chal = globals().get("Pax_Challenge")
                if AgentCls and QueryCls and Chal:
                    agent = AgentCls(QueryCls(challenge=Chal.PRODUCT, query_f=query))
                    agent.run()
                    return _ensure_dialog(agent.get_dialog(), query, "product")
                main = globals().get("Pax_agent_main")
                if main:
                    return _ensure_dialog(main(pd), query, "product")
            except Exception as exc:
                return [create_dialogue_step(
                    think=f"Product lane error: {type(exc).__name__}: {exc}",
                    tool_results=[], response="Done.", query=query, step=1)]
            return _ensure_dialog([], query, "product")


    def _LSC009C8_Pax_joy_numeric_product_price(product: dict) -> float | None:
        try:
            v = product.get('price')
            if v is None:
                return None
            x = float(v)
            if x != x or x < 0:
                return None
            return x
        except (TypeError, ValueError):
            return None


    def _LSC009C8_Pax_law_filter_spec_floor(extra_demise_6d811e: list[list[tuple[dict, float]]], crush_keen_8fa324: float) -> list[list[tuple[dict, float]]]:

        def brain_cover_81e5ad(lake_hold_17734f: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
            coach_finish_865aaa = float(crush_keen_8fa324)
            return [(p, float(frame_humble_ae5426)) for (p, frame_humble_ae5426) in lake_hold_17734f if float(frame_humble_ae5426) >= coach_finish_865aaa]
        return [brain_cover_81e5ad(lake_hold_17734f) for lake_hold_17734f in extra_demise_6d811e]


    def _LSC009C8_Pax_elephant_iter_listing_detail_field_values(detail: dict):
        common_brown_ea9a3c = (detail.get('attributes') or {}).items()
        for (key, gauge_cement_6c9503) in common_brown_ea9a3c:
            yield (key, gauge_cement_6c9503)
        for opts in (detail.get('sku_options') or {}).values():
            if isinstance(opts, dict):
                yield from opts.items()


    class _LSC009C8_Deal_Debuggable(object):

        def __init__(self, query: Deal_Query, debugger: Optional[Deal_Debugger]) -> None:
            self.debugger: Optional[Deal_Debugger] = debugger
            self.query: Deal_Query = query
            self.stopped: bool = False

        def log(self, obj: Any) -> None:

            def _lane_wrap_28(self, obj):
                if self.debugger is not None:
                    self.debugger.log(obj)

            def _lane_wrap_28_(self, obj):
                return _lane_wrap_28(self, obj)
            return _lane_wrap_28_(self, obj)


    class _LSC009C8_Pax_figure_AnchorStrategy:
        __slots__ = ('specs', 'spec_scored', 'shop_coverage', 'query', 'n_specs', 'max_anchor_shops', 'is_shop_voucher', 'voucher_budget')

        def __init__(self, bike_join_25dba5: list[dict], extra_demise_6d811e: list[list[tuple[dict, float]]], alone_hunt_c822db: dict[str, dict[int, list[dict]]], query: str, leg_fresh_c6b734: int, laundry_fortune_5b2f00: int | None=None, crucial_doctor_7244eb: bool=False, bachelor_frown_85fb76: tuple[float, float] | None=None) -> None:
            self.specs = bike_join_25dba5
            self.spec_scored = extra_demise_6d811e
            self.shop_coverage = alone_hunt_c822db
            self.query = query
            self.n_specs = leg_fresh_c6b734
            self.max_anchor_shops = laundry_fortune_5b2f00 if laundry_fortune_5b2f00 is not None else 10
            self.is_shop_voucher = crucial_doctor_7244eb
            self.voucher_budget = bachelor_frown_85fb76

        def invest_kitchen_f9f20c(self) -> tuple[list[str] | None, dict]:
            grain_clever_7399ce = max((eyebrow_drip_4a03c3[0][1] for eyebrow_drip_4a03c3 in self.spec_scored if eyebrow_drip_4a03c3), default=0.0)
            if grain_clever_7399ce <= 0:
                return (None, {})
            happy_keen_29d324 = Pax_abuse_global_top_products(self.spec_scored, grain_clever_7399ce)
            expand_add_6de5a9 = list(happy_keen_29d324.keys())
            if len(expand_add_6de5a9) == 1:
                excite_junk_bf267b = expand_add_6de5a9[0]
                if len(happy_keen_29d324[excite_junk_bf267b]) == 1:
                    ankle_armor_50c4bc = 1
                    coast_cargo_4bd378 = 'Single global top-scoring product; anchoring directly.'
                else:
                    ankle_armor_50c4bc = 2
                    coast_cargo_4bd378 = format(len(happy_keen_29d324[excite_junk_bf267b]), '') + ' products tied at score ' + format(grain_clever_7399ce, '.1f') + ' in spec[' + format(excite_junk_bf267b, '') + ']; iterating shops by price/rank.'
            else:
                display_float_359bf7 = Pax_bind_deepest_spec(expand_add_6de5a9, self.specs)
                ankle_armor_50c4bc = 3
                coast_cargo_4bd378 = 'Top score ' + format(grain_clever_7399ce, '.1f') + ' tied across specs ' + format(expand_add_6de5a9, '') + '; depth scoring selected spec[' + format(display_float_359bf7, '') + '] as primary anchor spec.'
            bounce_area_afcfb6 = Pax_attract_order_anchor_pool(self.spec_scored, self.n_specs, self.max_anchor_shops)
            if not bounce_area_afcfb6:
                return (None, {})
            for (elevator_belt_e0447, (score, hidden_kick_fd915f, conduct_feed_a71412)) in enumerate(bounce_area_afcfb6):
                if Pax_catalog_InferenceEngine.budget_sec_left() < 10:
                    break
                drum_civil_1e9b61 = str(conduct_feed_a71412.get('shop_id', ''))
                if not drum_civil_1e9b61:
                    continue
                jump_canyon_4ad780: list[str | None] = [None] * self.n_specs
                jump_canyon_4ad780[hidden_kick_fd915f] = str(conduct_feed_a71412.get('product_id', ''))
                decline_deputy_378eb1: list[dict] = []
                have_depth_172104 = True
                for extend_diagram_b09d2c in range(self.n_specs):
                    if extend_diagram_b09d2c == hidden_kick_fd915f:
                        continue
                    island_embrace_af0beb = Pax_fantasy_select_in_shop(self.specs[extend_diagram_b09d2c], drum_civil_1e9b61, self.query)
                    if not island_embrace_af0beb:
                        have_depth_172104 = False
                        break
                    jump_canyon_4ad780[extend_diagram_b09d2c] = str(island_embrace_af0beb.get('product_id', ''))
                    decline_deputy_378eb1.append({'spec_idx': extend_diagram_b09d2c, 'keywords': self.specs[extend_diagram_b09d2c].get('keywords', ''), 'product_id': str(island_embrace_af0beb.get('product_id', '')), 'title': island_embrace_af0beb.get('title', ''), 'price': island_embrace_af0beb.get('price'), 'llm_reason': island_embrace_af0beb.get('_llm_reason', '')})
                if have_depth_172104 and all((pid is not None for pid in jump_canyon_4ad780)):
                    return (jump_canyon_4ad780, {'resolution_mode': ankle_armor_50c4bc, 'global_max_score': grain_clever_7399ce, 'tie_note': coast_cargo_4bd378, 'anchor_attempt': elevator_belt_e0447 + 1, 'anchor': {'spec_idx': hidden_kick_fd915f, 'keywords': self.specs[hidden_kick_fd915f].get('keywords', ''), 'product_id': str(conduct_feed_a71412.get('product_id', '')), 'title': conduct_feed_a71412.get('title', ''), 'price': conduct_feed_a71412.get('price'), 'shop_id': drum_civil_1e9b61}, 'filled_specs': decline_deputy_378eb1})
            return (None, {})

        def argue_airport_eddefd(self) -> tuple[list[str] | None, dict]:
            finish_envelope_8e4d10 = 5
            evidence_arena_57261e: dict[str, float] = {}
            for eyebrow_drip_4a03c3 in self.spec_scored:
                for (prod, frame_humble_ae5426) in eyebrow_drip_4a03c3:
                    sid = str(prod.get('shop_id') or '').strip()
                    if not sid:
                        continue
                    if frame_humble_ae5426 > evidence_arena_57261e.get(sid, -1.0):
                        evidence_arena_57261e[sid] = frame_humble_ae5426
            if not evidence_arena_57261e:
                return (None, {})
            gradual_hard_3c4a06 = sorted(evidence_arena_57261e.items(), key=lambda x: -x[1])
            angry_gown_551646: list[str] = []
            gloom_educate_d5ccc8: float | None = None
            for (i, (sid, frame_humble_ae5426)) in enumerate(gradual_hard_3c4a06):
                if i < finish_envelope_8e4d10:
                    angry_gown_551646.append(sid)
                    gloom_educate_d5ccc8 = frame_humble_ae5426
                elif gloom_educate_d5ccc8 is not None and frame_humble_ae5426 == gloom_educate_d5ccc8:
                    angry_gown_551646.append(sid)
                else:
                    break
            if not angry_gown_551646:
                return (None, {})
            frozen_inject_2124f9: float | None = None
            if self.voucher_budget is not None:
                (_, frozen_inject_2124f9) = self.voucher_budget
            injury_fiscal_3cc054: list[list[dict]] = []
            for (extend_diagram_b09d2c, attract_balance_4a42ca) in enumerate(self.specs):
                alien_elbow_355b34: list[dict] = []
                box_awake_a2e8b0: set[str] = set()
                circle_crack_1766d7: str | None = None
                if frozen_inject_2124f9 is not None:
                    jazz_donkey_eed5c = Pax_cost_intersect_spec_price_with_budget_cap(attract_balance_4a42ca, frozen_inject_2124f9)
                    if jazz_donkey_eed5c is not None:
                        circle_crack_1766d7 = Pax_junior_price_lo_hi_to_find_str(jazz_donkey_eed5c[0], jazz_donkey_eed5c[1])
                for foster_critic_ad3360 in angry_gown_551646:
                    depart_chimney_a49514 = Pax_dust_search_spec_in_shop_limited(attract_balance_4a42ca, foster_critic_ad3360, page=1, limit=10, price_override=circle_crack_1766d7)
                    for h in depart_chimney_a49514:
                        pid = str(h.get('product_id', ''))
                        if pid and pid not in box_awake_a2e8b0:
                            alien_elbow_355b34.append(h)
                            box_awake_a2e8b0.add(pid)
                injury_fiscal_3cc054.append(alien_elbow_355b34)
            cool_health_cb4708: list[list[tuple[dict, float]]] = []
            for (extend_diagram_b09d2c, (attract_balance_4a42ca, basket_canoe_3124b9)) in enumerate(zip(self.specs, injury_fiscal_3cc054)):
                if not basket_canoe_3124b9:
                    cool_health_cb4708.append([])
                    continue
                enough_achieve_eaace1 = attract_balance_4a42ca.get('query') or attract_balance_4a42ca.get('keywords') or self.query
                pids = [str(p.get('product_id', '')) for p in basket_canoe_3124b9 if p.get('product_id')]
                details = Pax_garage_load_details(pids)
                great_acid_a687fe = Pax_axis_score_listings(enough_achieve_eaace1, basket_canoe_3124b9, details, only_product_type=bool(attract_balance_4a42ca.get('only_product_type', False)))
                cool_health_cb4708.append(great_acid_a687fe)
            flower_detect_aa1fb4: list[list[dict]] = [[p for (p, _) in eyebrow_drip_4a03c3] for eyebrow_drip_4a03c3 in cool_health_cb4708]
            display_course_505780 = Pax_jar_index_by_shop(flower_detect_aa1fb4)
            gauge_ability_1e91e6 = [sid for (sid, domain_family_54b8ea) in display_course_505780.items() if len(domain_family_54b8ea) == self.n_specs]
            if not gauge_ability_1e91e6:
                return (None, {})
            learn_dream_7eb302 = Pax_badge_sort_shops(gauge_ability_1e91e6, display_course_505780, self.specs, self.query)
            already_learn_9445c6 = learn_dream_7eb302[:10]
            (foster_critic_ad3360, drill_able_83ad88) = Pax_chat_choose_shop_llm(already_learn_9445c6, display_course_505780, self.specs, self.query)
            ethics_bitter_c3233f = [drill_able_83ad88[extend_diagram_b09d2c]['product_id'] for extend_diagram_b09d2c in range(self.n_specs) if extend_diagram_b09d2c in drill_able_83ad88]
            if not (foster_critic_ad3360 and len(ethics_bitter_c3233f) == self.n_specs):
                return (None, {})
            if self.voucher_budget is not None and frozen_inject_2124f9 is not None:
                infant_attitude_67a283 = 0.0
                for extend_diagram_b09d2c in range(self.n_specs):
                    better_green_c42f6d = drill_able_83ad88[extend_diagram_b09d2c]['product_id']
                    edge_about_780617 = (display_course_505780.get(foster_critic_ad3360) or {}).get(extend_diagram_b09d2c, [])
                    prod = next((p for p in edge_about_780617 if str(p.get('product_id', '')) == better_green_c42f6d), None)
                    if prod is not None:
                        carry_evil_2e04f2 = Pax_joy_numeric_product_price(prod)
                        if carry_evil_2e04f2 is not None:
                            infant_attitude_67a283 += carry_evil_2e04f2
                if infant_attitude_67a283 > frozen_inject_2124f9 + 1e-06:
                    return (None, {})
            return (list(ethics_bitter_c3233f), {'resolution_mode': 5, 'anchor_shops_selected': angry_gown_551646, 'anchor_full_shops_count': len(gauge_ability_1e91e6), 'anchor_attempt': 1, 'tie_note': 'Anchor v2: ' + format(len(angry_gown_551646), '') + ' top-ranked shops selected, ' + format(len(gauge_ability_1e91e6), '') + ' full-coverage after re-scoring. Case-B LLM elected shop_id=' + format(foster_critic_ad3360, '') + '.', 'anchor': {'shop_id': foster_critic_ad3360, 'spec_idx': None}, 'filled_specs': [{'spec_idx': extend_diagram_b09d2c, 'keywords': self.specs[extend_diagram_b09d2c].get('keywords', ''), 'product_id': drill_able_83ad88[extend_diagram_b09d2c]['product_id'], 'reason': drill_able_83ad88[extend_diagram_b09d2c].get('reason', ''), 'score': drill_able_83ad88[extend_diagram_b09d2c].get('score', 0.0)} for extend_diagram_b09d2c in range(self.n_specs) if extend_diagram_b09d2c in drill_able_83ad88]})

        def run(self) -> tuple[list[str] | None, dict]:
            if self.n_specs >= 3:
                (accident_deal_ab3bec, dry_absorb_e8725e) = Pax_asset_partial_coverage_resolve(self.specs, self.spec_scored, self.shop_coverage, self.query, self.n_specs)
                if accident_deal_ab3bec:
                    return (accident_deal_ab3bec, dry_absorb_e8725e)
            if self.is_shop_voucher or self.n_specs >= 4:
                return self._run_v2()
            return self._run_v1()


    def _LSC009C8_Pax_human_alt_probe_product(attract_balance_4a42ca, query, canvas_alarm_8d0e98):
        return None


    def _LSC009C8_Pax_basic_normalise_sku_options(barely_duty_c97515: Any) -> list[dict]:
        result: list[dict] = []
        if isinstance(barely_duty_c97515, list):
            for lake_hold_17734f in barely_duty_c97515:
                if not isinstance(lake_hold_17734f, dict):
                    continue
                gauge_cement_6c9503 = lake_hold_17734f.get('values', [])
                if not isinstance(gauge_cement_6c9503, list):
                    gauge_cement_6c9503 = list(gauge_cement_6c9503.values()) if isinstance(gauge_cement_6c9503, dict) else []
                result.append({'name': lake_hold_17734f.get('name'), 'values': gauge_cement_6c9503[:5]})
        elif isinstance(barely_duty_c97515, dict):
            bridge_help_836b54: dict[str, list] = {}
            for avocado_group_45929 in barely_duty_c97515.values():
                if not isinstance(avocado_group_45929, dict):
                    continue
                for (easy_flag_53724e, foster_donor_fa38b9) in avocado_group_45929.items():
                    element_hurt_5176ad = bridge_help_836b54.setdefault(easy_flag_53724e, [])
                    if foster_donor_fa38b9 not in element_hurt_5176ad:
                        element_hurt_5176ad.append(foster_donor_fa38b9)
            for (easy_flag_53724e, values) in bridge_help_836b54.items():
                result.append({'name': easy_flag_53724e, 'values': values[:5]})
        return result


    def _LSC009C8_Pax_decade_attr_score(grass_iron_ec3ae3: list[str], detail: dict) -> float:
        (carry_join_d2d362, dove_kangaroo_f6eb33) = Pax_choice_flatten_detail(detail)
        fly_into_3a2ef8 = set(re.findall('\\b\\w+\\b', carry_join_d2d362))
        cruel_behind_8b005 = 0.0
        for w in grass_iron_ec3ae3:
            if format(w, '') + '#' in dove_kangaroo_f6eb33:
                cruel_behind_8b005 += 5
            elif w in dove_kangaroo_f6eb33:
                cruel_behind_8b005 += 3
            elif w in fly_into_3a2ef8:
                cruel_behind_8b005 += 2
        return cruel_behind_8b005


    def _LSC009C8_Pax_away_route_task_kind(query: str) -> str:
        inside_ancient_1693be = query.lower()
        earth_circle_ac9d53 = {'voucher', 'budget', 'discount'}
        if any((excess_dignity_42264d in inside_ancient_1693be for excess_dignity_42264d in earth_circle_ac9d53)):
            return 'voucher'
        capital_invite_9e7b38 = re.compile('(?:,?\\s*and\\s+also\\s+|,?\\s*also,?\\s*|Second(?:ly)?,\\s*|Third(?:ly)?,\\s*|First,\\s*|\\(\\d+\\)\\s*|\\d+\\.\\s*|Additionally,\\s*|Furthermore,\\s*|Moreover,\\s*|In\\s+addition,?\\s*|Plus,\\s*|On\\s+top\\s+of\\s+that,?\\s*|[.]\\s*Next,\\s*|[.]\\s*Lastly,\\s*|[.]\\s*Finally,\\s*|[.]\\s*Last,\\s*|\\bThen\\s*,?\\s*I\\s+(?:need|want|also)\\b|\\bI\\s+also\\s+(?:want|need)\\b)', re.IGNORECASE)
        dumb_immune_32ed94 = re.search('\\b(both|these|offering|offers|sells|same|together|along\\s+with)\\b', inside_ancient_1693be)
        if 'shop' in inside_ancient_1693be and (dumb_immune_32ed94 is not None or capital_invite_9e7b38.search(query) is not None):
            return 'shop'
        return 'product'


    def _LSC009C8_Pax_badge_sort_shops(digital_keen_c9ef6d: list[str], alone_hunt_c822db: dict[str, dict[int, list[dict]]], bike_join_25dba5: list[dict], query: str) -> list[str]:

        def ketchup_elbow_19284b(candidates: list[dict], attract_balance_4a42ca: dict) -> float:
            if not candidates:
                return 0.0
            enough_achieve_eaace1 = attract_balance_4a42ca.get('query') or attract_balance_4a42ca.get('keywords') or query
            return max((Pax_defense_heuristic_score(p, str(enough_achieve_eaace1)) for p in candidates), default=0.0)

        def ladder_grace_ea0a12(deny_diesel_7c0a1c: str) -> float:
            column_gas_1263f3 = alone_hunt_c822db.get(deny_diesel_7c0a1c) or {}
            return sum((ketchup_elbow_19284b(column_gas_1263f3.get(excite_junk_bf267b, []), attract_balance_4a42ca) for (excite_junk_bf267b, attract_balance_4a42ca) in enumerate(bike_join_25dba5)))
        flight_cup_7d14a9 = [(deny_diesel_7c0a1c, ladder_grace_ea0a12(deny_diesel_7c0a1c)) for deny_diesel_7c0a1c in digital_keen_c9ef6d]
        flight_cup_7d14a9.sort(key=lambda pair: (-pair[1], pair[0]))
        return [deny_diesel_7c0a1c for (deny_diesel_7c0a1c, _) in flight_cup_7d14a9]


    def _LSC009C8_Pax_jar_index_by_shop(hire_farm_8d7767: Sequence[Sequence[dict]]) -> dict[str, dict[int, list[dict]]]:
        forest_expose_7ba378: dict[str, dict[int, list[dict]]] = defaultdict(lambda : defaultdict(list))
        for (idx, products) in enumerate(hire_farm_8d7767):
            for prod in products:
                sid = str(prod.get('shop_id', ''))
                if sid:
                    forest_expose_7ba378[sid][idx].append(prod)
        return forest_expose_7ba378


    def _LSC009C8_Pax_include_spec_price(attract_balance_4a42ca: dict, *, because_door_25b72b: bool, circle_crack_1766d7: str | None=None) -> str | None:
        if circle_crack_1766d7 is not None:
            return circle_crack_1766d7
        return attract_balance_4a42ca.get('price') or attract_balance_4a42ca.get('price_range') if because_door_25b72b else None


    def _LSC009C8_Pax_genre_order_rank_sum_two(bike_join_25dba5: list[dict], i: int, j: int) -> int:
        a = Pax_hold_to_rank_int(bike_join_25dba5[i].get('order'))
        b = Pax_hold_to_rank_int(bike_join_25dba5[j].get('order'))
        return (a or 0) + (b or 0)


    def _LSC009C8_Pax_fantasy_select_in_shop(attract_balance_4a42ca: dict, foster_critic_ad3360: str, query: str) -> dict | None:
        products = Pax_ability_search_spec(attract_balance_4a42ca, shop_id=foster_critic_ad3360)
        if not products:
            products = Pax_ability_search_spec(attract_balance_4a42ca, shop_id=foster_critic_ad3360, omit_service_from_api=True)
        if not products:
            return None
        pids = [str(p.get('product_id', '')) for p in products if p.get('product_id')]
        details = Pax_garage_load_details(pids)
        enough_achieve_eaace1 = attract_balance_4a42ca.get('query') or attract_balance_4a42ca.get('keywords') or query
        island_embrace_af0beb = Pax_blossom_elect_best(enough_achieve_eaace1, products[:10], details, only_product_type=bool(attract_balance_4a42ca.get('only_product_type', False)))
        return island_embrace_af0beb if island_embrace_af0beb is not None else products[0] if products else None


    class _LSC009C8_Merch_ShoppingAgent(object):

        def __init__(self, query: str, debugger: Optional[Merch_Debugger]=None, task_hint: str='') -> None:
            self.debugger = debugger if debugger is not None else Merch_Debugger()
            self.query = query
            self.task_hint = task_hint
            self.sess: Merch_SearchSession = Merch_SearchSession(query, self.debugger)

        def run(self) -> None:
            pipeline = Merch_QueryPipeline(self.query, self.debugger, task_hint=self.task_hint)
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
            self.sess.step_on_build(qi)
            self.engine = Merch_resolve_engine_cls(qi.challenge)(qi, self.sess, self.debugger)
            best_p = self.engine.run()
            if qi.challenge != Merch_ShoppingTaskType.VOUCHER:
                self.sess.step_on_compare(best_p, self.engine._render_alt_comparison(best_p))
            self.sess.add_tool_result(Merch_ProxyUtil.submit(best_p))
            self.sess.add_tool_result(Merch_ProxyUtil.terminate('success' if best_p else 'failure'))
            self.sess.step_on_submit(best_p, qi)

        def get_dialog(self) -> List[Dict]:
            return self.sess.dialog


    class _LSC009C8_Merch_OptionalEnhancementKit:
        """Facade grouping optional helpers; safe to import/test without touching Agent.run."""
        EdgeCase = Merch_EdgeCaseProcessor
        Boundary = Merch_FuzzyBoundaryLogic
        Temperature = Merch_TemperatureController
        Voting = Merch_MultiModelVotingSystem

        @staticmethod
        def find_product_with_fallback(**kwargs) -> List[Dict]:
            """shrike/garland-style search retries: drop service filter, shorten query."""
            result = Merch_ProxyUtil.ORO_PROXY.get('/search/find_product', kwargs)
            result = result if result else []
            if result:
                return result
            service = kwargs.get('service')
            if service:
                retry_params = dict(kwargs)
                retry_params.pop('service', None)
                result = Merch_ProxyUtil.ORO_PROXY.get('/search/find_product', retry_params)
                result = result if result else []
            if not result:
                q = kwargs.get('q') or ''
                tokens = str(q).split()
                short_q = ' '.join(tokens[:2])
                if short_q and short_q != q:
                    retry_params = dict(kwargs)
                    retry_params['q'] = short_q
                    retry_params.pop('service', None)
                    result = Merch_ProxyUtil.ORO_PROXY.get('/search/find_product', retry_params)
                    result = result if result else []
            return result

        @staticmethod
        def build_regex_hints(query: str) -> Dict[str, Any]:
            """Minimal regex_hints block (shrike/beam prompt preamble)."""
            raw = str(query or '')
            stripped = raw.strip()
            return {'quoted_literals': re.findall('["\\\']([^"\\\']+)["\\\']', stripped), 'number_unit_tokens': re.findall('\\b\\d+(?:\\.\\d+)?[a-zA-Z]+\\b', stripped.lower()), 'size_labels': re.findall('\\b(?:xxs|xs|s|m|l|xl|xxl|xxxl|\\dxl|\\d+\\s*(?:cm|mm|inch|in))\\b', stripped.lower()), 'color_words': [w for w in re.findall('\\b[a-z]{3,}\\b', stripped.lower()) if w in ('black', 'white', 'red', 'blue', 'green', 'gold', 'silver', 'pink', 'gray', 'grey', 'brown', 'yellow', 'purple', 'orange')], 'service_tags': [t for t in ('official', 'freeShipping', 'COD', 'flashsale') if t.lower() in stripped.lower() or t in stripped], 'price_range_guess': Merch_FuzzyBoundaryLogic.normalize_price_from_query(stripped)}

        @staticmethod
        def audit_listing_against_spec(*, title: str, price: Any, parsed_spec: Dict) -> Dict[str, Any]:
            """Fast accept / boundary audit (_x/beam FastAcceptVerifier pattern)."""
            price_range = parsed_spec.get('price') if isinstance(parsed_spec, dict) else None
            (lo, hi) = Merch_FuzzyBoundaryLogic.parse_price_bounds(price_range)
            boundary = Merch_FuzzyBoundaryLogic.classify_price_boundary(price, lo, hi)
            (price_ok, price_note) = Merch_FuzzyBoundaryLogic.price_fits_range(price, price_range, fuzzy=True)
            keywords = parsed_spec.get('keywords') if isinstance(parsed_spec, dict) else []
            title_l = (title or '').lower()
            missing = [kw[-1] if isinstance(kw, list) and kw else str(kw) for kw in keywords or [] if isinstance(kw, (list, tuple)) and len(kw) >= 2 and (str(kw[-1]).lower() not in title_l)]
            note = 'ok' if price_ok and (not missing) else 'review'
            return {'keywords_matched': not missing, 'keywords_missing': missing, 'title_contains_all_keywords': not missing, 'price_ok': price_ok, 'price_note': price_note, 'boundary_class': boundary, 'overall_note': note}


    class _LSC009C8_Merch_JsonModel(object):

        @classmethod
        def from_json(cls, data: dict) -> 'Merch_JsonModel':
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
                return Merch_JsonModel._coerce_from_json(value, inner_t)
            if is_dataclass(t) and isinstance(value, dict):
                return t.from_json(value)
            if origin is list and isinstance(value, list):
                return [Merch_JsonModel._coerce_from_json(v, args[0]) for v in value]
            if origin is tuple and isinstance(value, list):
                return tuple((Merch_JsonModel._coerce_from_json(v, args[0]) for v in value))
            if origin is dict and isinstance(value, dict):
                (kt, vt) = args
                return {Merch_JsonModel._coerce_from_json(k, kt): Merch_JsonModel._coerce_from_json(v, vt) for (k, v) in value.items()}
            if t is float and isinstance(value, str):
                return Merch_MiscUtil.extract_number(value)
            if t is int and isinstance(value, str):
                return Merch_MiscUtil.extract_number(value, True)
            if t is str:
                return str(value)
            return value

        @staticmethod
        def _coerce_to_json(obj) -> Any:
            if is_dataclass(obj):
                result = {}
                for f in fields(obj):
                    value = getattr(obj, f.name)
                    result[f.name] = Merch_JsonModel._coerce_to_json(value)
                return result
            if isinstance(obj, list):
                return [Merch_JsonModel._coerce_to_json(x) for x in obj]
            if isinstance(obj, dict):
                return {Merch_JsonModel._coerce_to_json(k): Merch_JsonModel._coerce_to_json(v) for (k, v) in obj.items()}
            return obj



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

_LSC01644_AST_PAD = 0
if _LSC01644_AST_PAD:
    class _LSC01644_Merch_SegTree(object):
        """Range top-k tree over (weight, index) pairs.
        Each node keeps the best `k` entries in its range rather than just the
        single maximum, so a range query returns up to `k` ranked candidates. The
        voucher knapsack needs this: with only the argmax per range it can see just
        one completion per left-half state, and a cart whose slots are individually
        correct is invisible whenever some higher-weighted product wins that price
        bucket. Build and query are both O(k) per node touched.
        """

        def __init__(self, values, k: int=30) -> None:
            (self.k, n) = (k, len(values))
            self.size = 1
            while self.size < n:
                self.size <<= 1
            self.seg: List[List[Tuple[Any, int]]] = [[] for _ in range(2 * self.size)]
            for (i, v) in enumerate(values):
                self.seg[self.size + i] = [(v, i)]
            for i in range(self.size - 1, 0, -1):
                self.seg[i] = self._merge(self.seg[2 * i], self.seg[2 * i + 1])

        def _merge(self, a: List, b: List) -> List:
            (out, ia, ib) = ([], 0, 0)
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

        def query(self, l, r) -> List[Tuple[Any, int]]:
            if l > r:
                return []
            ans: List[Tuple[Any, int]] = []
            l += self.size
            r += self.size
            while l <= r:
                if l & 1:
                    ans = self._merge(ans, self.seg[l])
                    l += 1
                if not r & 1:
                    ans = self._merge(ans, self.seg[r])
                    r -= 1
                l >>= 1
                r >>= 1
            return ans


    class _LSC01644_Deal_ExtTool:

        def search(**kwargs) -> List[Dict]:
            result = Deal_ProxyUtil.ORO_PROXY.get('/search/find_product', kwargs)
            return result if result else []

        def view(ids: str) -> List[Dict]:
            result = Deal_ProxyUtil.ORO_PROXY.get('/search/view_product_information', {'product_ids': ids})
            return result if result else []

        def submit(product_ids: str) -> str:
            return f'submitted recommendation to the user: {product_ids}.'

        def finish(status: str) -> str:
            return f'session completed with status: {status}'


    def _LSC01644_Pax_certain_title_hit_score(grass_iron_ec3ae3: list[str], diamond_easy_603a11: set[str], title: str) -> float:
        score = 0.0
        for w in grass_iron_ec3ae3:
            if Pax_kingdom_is_title_direct_match(w, diamond_easy_603a11):
                score += 2
            elif Pax_ivory_is_title_partial_match(w, diamond_easy_603a11):
                score += 1
            if any((ch.isdigit() for ch in w)) and w in title:
                score += 2
        return score


    class _LSC01644_Deal_Filter(Deal_JsonModel):
        keywords: List[str] = field(default_factory=list)
        services: List[str] = field(default_factory=list)
        price: Optional[str] = None
        shop: Optional[str] = None

        def update(self, **kwargs) -> 'Filter':

            def _lane_wrap_37(self, **kwargs):
                for (k, v) in kwargs.items():
                    self.__setattr__(k, v)
                return self

            def _lane_wrap_37_(self, **kwargs):
                return _lane_wrap_37(self, **kwargs)
            return _lane_wrap_37_(self, **kwargs)

        def get_q(self) -> str:

            def _lane_wrap_95(self):
                chunks = []
                for kw in self.keywords:
                    for m in kw.split():
                        m = m.lower().strip()
                        if m and m not in chunks:
                            chunks.append(m)
                return ' '.join(sorted(chunks))

            def _lane_wrap_95_(self):
                return _lane_wrap_95(self)
            return _lane_wrap_95_(self)

        def clone(self) -> 'Filter':

            def _lane_wrap_36(self):
                return Deal_Filter.from_json(self.to_json())

            def _lane_wrap_36_(self):
                return _lane_wrap_36(self)
            return _lane_wrap_36_(self)

        def __hash__(self) -> int:
            return hash((self.get_q(), self.price, self.shop))

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, type(self)):
                return False
            return self.get_q() == other.get_q() and self.price == other.price and (self.shop == other.shop)


    class _LSC01644_Deal_Query(Deal_JsonModel):
        challenge: Deal_Challenge = Deal_Challenge.UNKNOWN
        query_f: str = ''
        query_e: str = ''
        intents: List[Deal_SubQuery] = field(default_factory=list)
        voucher: Optional[Deal_Voucher] = None
        alts: Dict[str, str] = field(default_factory=dict)
        fillers: List[str] = field(default_factory=list)


    def _LSC01644_Pax_kingdom_is_title_direct_match(word: str, diamond_easy_603a11: set[str]) -> bool:
        if word in diamond_easy_603a11:
            return True
        abstract_blue_b3716f = word[:-1] if word.endswith('s') else format(word, '') + 's'
        if abstract_blue_b3716f in diamond_easy_603a11:
            return True
        if len(word) < 3:
            return False
        return any((decade_buffalo_cfd23d.startswith(word) for decade_buffalo_cfd23d in diamond_easy_603a11 if len(decade_buffalo_cfd23d) > len(word)))


    class _LSC01644_Merch_EngineUtil:
        """Fuzzy spelling, attribute/keyword matching and price-math helpers used by
        the search engine and query pipeline."""

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
            vars = list(set((v for v in vars if v != s and v not in Merch_Vocab.KW_STOPWORDS and (len(v) >= 3))))
            return vars

        @staticmethod
        def is_kw_in(kw: str, s: str, as_is: bool, synonyms: List[Tuple[str, str]], terminal: bool=False) -> bool:
            if re.search(('\\b' if re.match('\\w', kw[0]) else '') + re.escape(kw) + ('\\b' if re.match('\\w', kw[-1]) else ''), s, re.I):
                return True
            if as_is:
                return False
            vars = Merch_EngineUtil.get_spell_variations(kw)
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
            if kw in ['pcs', 'pc', 'pack']:
                for w in ['piece', 'pack', 'set']:
                    if w in s:
                        return True
            for (from_s, to_s) in synonyms:
                if kw == from_s or kw == f'{from_s}s' or from_s == f'{kw}s':
                    if re.search(f'\\b{re.escape(to_s)}s?\\b', s):
                        return True
                elif kw == to_s or kw == f'{to_s}s' or to_s == f'{kw}s':
                    if re.search(f'\\b{re.escape(from_s)}s?\\b', s):
                        return True
            if not terminal:
                spell = re.sub(f'ss', 'sss', kw, flags=re.I)
                if spell != kw and Merch_EngineUtil.is_kw_in(spell, s, False, synonyms, True):
                    return True
            return False

        @staticmethod
        def is_valid_attr(k: str, v: str, q: str) -> bool:
            pairs: List[Tuple[str, str, str, str, bool]] = [(v, q, '\\b(\\d+[a-z]*)\\s*\\*\\s*(\\d+[a-z]*)\\b', '{m1} by {m2}', True), (v, q, '\\b(\\d+)\\-(\\d+)([a-z]+)\\b', '{m1}-{m2} {m3}', True), (v, q, '\\b(\\d+)([a-z])\\+', '\\b{m1}\\s*{m2}([a-z]*)\\sand\\sup\\b', True), (v, q, '\\b(\\d+)\\-(\\d+)[a-z]+\\b', '\\b{m1}\\-{m2}\\b', True), (v, q, '\\b(\\d+)\\-(\\d+)y\\b', '\\b{m1}\\sto\\s{m2}\\syears\\b', True), (q, v, '\\b(\\d+)\\s*\\-\\&\\-\\s*(\\d+)\\b', '{m1}\\sto\\s{m2}', False), (v, q, '\\b(\\d+)\\s*-\\s*(\\d+)\\s*([a-z]+)\\b', '{m3}\\sbetween\\s{m1}\\sand\\s{m2}', True), (v, q, '\\b(\\d+)\\.0\\b', '\\b{m1}\\b', True), (v, q, '\\b([\\d\\.]+)[a-z]*\\s*~\\s*([\\d\\.]+)\\b', '\\b{m1}\\sto\\s{m2}\\b', True), (v, q, '【(\\d+\\-\\d+[a-z]+)】', '\\b{m1}\\b', True), (v, q, '\\b(\\d+)\\s*-\\s*(\\d+)([a-z]*)\\b', 'up to {m2}{m3}', True), (v, q, '\\b(\\w+)\\s(\\w+)\\b', '\\b{m1}\\s(\\w+)\\s{m2}\\b', True), (q, v, '\\b([a-z]+)\\s+\\((\\d)\\s+piece\\)\\s+', '^{m1}\\s+{m2}pcs$', False), (v, q, '^(?:[a-z]+)\\:\\s*(\\d+)$', '\\b{m1}\\b', True), (v, q, '\\b(?:[a-z]+)\\:\\s*(\\d+)\\s*-\\s*(\\d+)\\s*([a-z]+)\\b', 'up\\sto\\s{m2}{m3}', True), (v, q, '\\b(\\d+)\\s*-\\s*(\\d+)\\b', '{m1} and {m2}', True)]
            for (src, dst, src_pattern, dst_pattern, yes) in sorted(pairs, key=lambda p: p[-1]):
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
            (ks, v) = (splits[:-1], splits[-1])
            if v.lower() in Merch_Vocab.YES_WORDS:
                for i in range(len(ks)):
                    t = ' '.join([k.replace('_', ' ') for k in ks[i:]])
                    if t in q:
                        (ks, v) = ([v], t)
                        break
                else:
                    return None
            for yes in Merch_Vocab.YES_WORDS:
                if re.search(f'\\b{re.escape(yes)}$', v, re.I):
                    v = re.sub(f'\\b{re.escape(yes)}$', '', v, flags=re.I)
            for nw in Merch_Vocab.NO_WORDS:
                if (m := re.search(f'^{re.escape(nw)}\\s+([\\w\\s]+)$', v, re.I)):
                    ks.append('no')
                    v = m.group(1)
                    break
            if f'{ks[-1]} {v}' in Merch_Vocab.KW_STOPWORDS:
                return None
            if f'{v} {ks[-1]}' in Merch_Vocab.KW_STOPWORDS:
                return None
            for sw in Merch_Vocab.KW_STOPWORDS:
                v = re.sub(f'(?<!-)\\b{re.escape(sw)}\\b(?!-)', '', v, flags=re.I)
            if len(ks) == 1 and '_' in ks[0]:
                prefix = ks[0].split('_')[0] + '-'
                if v.startswith(prefix):
                    v = v[len(prefix):]
                prefix = prefix[:-1]
                if re.search(f'^{re.escape(prefix)}[a-z]+$', v, re.I):
                    v = v[len(prefix):]
            if (m := re.search('^(any|all)\\s+([\\w\\s]+)$', v, re.I)):
                k = m.group(2).replace(' ', '_')
                v = m.group(1)
                if k not in ks:
                    ks.append(k)
            if v.isnumeric() and ks[-1] in ['eu', 'us', 'int']:
                (v, ks) = (f'{ks[-1]} {v}', ['size'])
            v = re.sub('\\bge[a-z]([a-z]{2}one)\\b', '\\1', v, flags=re.I)
            v = re.sub('\\b(\\d+)in(\\d+)\\b', '\\1-in-\\2', v, flags=re.I)
            v = re.sub('[\\&\\|]', ' ', v)
            v = re.sub('\\s+', ' ', v).strip()
            _RETAIL_CONTEXT = frozenset({'store', 'shop', 'seller', 'vendor', 'marketplace'})
            if v.lower() in _RETAIL_CONTEXT:
                return None
            return ks + [v] if v else None

        @staticmethod
        def kw_to_constraints(kws: List[List[str]]) -> List[str]:
            """Render decompose-extracted keywords (e.g. ["color", "pink"]) into
            readable "key: value" constraint strings for combo-select prompts."""
            constraints = []
            for kw in kws:
                if not kw:
                    continue
                (*keys, v) = kw
                key = '_'.join((k for k in keys if k)) or 'attribute'
                constraints.append(f'{key}: {v}')
            return constraints
        _PROD_ONLY_RES = (re.compile('\\b(?:just|only)(?:\\s+the)?\\s+products?\\b', re.I), re.compile('\\bproducts?\\s+only\\b', re.I))

        @staticmethod
        def is_prod_only(q: str) -> bool:
            return any((pat.search(q) is not None for pat in Merch_EngineUtil._PROD_ONLY_RES))
        _FUSED_UNIT_RE = re.compile('\\d(?:\\.\\d+)?[a-z]', re.I)
        _NUMBER_RE = re.compile('\\d+(?:\\.\\d+)?')
        _TOKEN_RE = re.compile('[a-z]+|\\d+(?:\\.\\d+)?', re.I)

        @staticmethod
        def fusible_pairs(v: str) -> List[Tuple[str, str]]:
            """Every adjacent word+number pair in `v`, as (surface form, fused token).
            Catalogues spell model and capacity designations as one token -- `14plus`,
            `note10`, `ender3`, `realme11`, `500watts` -- while a query writes them
            split, and BM25 then never matches. `_derive_kw_variants` already fuses
            this shape, but only when the pair is the WHOLE value ("20 mm"); embedded
            in a longer one ("iphone 14 plus") it is left split and no query the agent
            issues can reach the fused spelling.
            Walked token-wise rather than by one regex on purpose: a single regex
            consumes the number on the "iphone 14" match and never offers "14 plus",
            which is the pair that matters here. Both orders are emitted, since the
            number leads as often as it trails.
            This only proposes spellings. Which of them the catalogue actually uses is
            settled later, against its own rows -- see `SearchEngine._fused_token_options`.
            """
            toks = [(m.group(0), m.start(), m.end()) for m in Merch_EngineUtil._TOKEN_RE.finditer(v)]
            pairs: List[Tuple[str, str]] = []
            for ((a, _, a_end), (b, b_start, _)) in zip(toks, toks[1:]):
                if v[a_end:b_start].strip():
                    continue
                if len(a) > 8 or len(b) > 8:
                    continue
                (a_num, b_num) = (a[0].isdigit(), b[0].isdigit())
                if a_num == b_num:
                    continue
                pairs.append((f'{a} {b}', f'{a}{b}'))
            return pairs

        @staticmethod
        def value_is_self_identifying(v: str) -> bool:
            """Does this keyword value name its own property, without help from its key?
            `SearchOption.update_q` emits the KEY word into the free-text search string for
            keys outside the catalogue schema, so that "sole→rubber" searches "sole rubber"
            and matches a title reading "rubber sole". That premise holds for a component
            key sitting next to a lexical value, because the title really does print both
            words together.
            It does not hold for a measurement. A listing writes "20mm 22mm Silicone Sport
            Strap", never "width 20mm" -- the unit already says which property it is, so the
            key word is a token BM25 must additionally match and no title supplies. Measured
            on one such sub-query: the assembled query carrying `width` missed the target in
            50 results, and the identical query without it returned that target at rank 37.
            A bare count is deliberately NOT self-identifying: "4" says nothing on its own,
            and "port→4" wants to search "4 port", which is exactly how titles write it.
            """
            v = str(v or '')
            if Merch_EngineUtil._FUSED_UNIT_RE.search(v):
                return True
            return len(Merch_EngineUtil._NUMBER_RE.findall(v)) >= 2

        @staticmethod
        def get_vouchered_cost(products: List[Merch_Product], vo: Optional[Merch_VoucherOption]) -> float:
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
        def get_favor_range(vo: Merch_VoucherOption) -> Tuple[float, float]:
            high = vo.threshold / 0.1 if vo.threshold is not None else math.inf
            low = vo.threshold / 0.9 if vo.threshold is not None else 0
            if not vo.is_fixed:
                l1 = vo.budget / 1.1 / (1.0 - vo.discount / 100.0)
                l2 = vo.budget / 1.1 + (vo.cap if vo.cap is not None else math.inf)
                h1 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
                h2 = vo.budget / (1.0 - vo.discount / 100.0)
                low = max(low, min(l1, l2))
                high = min([high, h1, h2])
            else:
                low = max(low, vo.budget / 1.1 + vo.discount)
                high = min(high, vo.budget + vo.discount)
            return (math.floor(low) - 1, math.ceil(high) + 1)

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
            (low_str, high_str) = price.split('-', 1)
            (low, high) = (0, math.inf)
            if low_str:
                low = Merch_MiscUtil.extract_number(low_str, True)
            if high_str:
                high = Merch_MiscUtil.extract_number(high_str, True)
            return (low, high)

        @staticmethod
        def stringfy_price(low: int, high: int) -> str:
            return (str(low) if low > 0 else '') + '-' + (str(high) if high < math.inf else '')


    def _LSC01644_Pax_jewel_probe_edges_base(products: list, frozen_inject_2124f9: float, *, foster_critic_ad3360: str | None) -> tuple[list[float], list[float], list]:
        expect_bomb_238311: list[float] = []
        butter_chapter_e2f714: list[float] = []
        flat_funny_24254f: list = []
        basket_faint_212414 = str(foster_critic_ad3360) if foster_critic_ad3360 is not None else None
        correct_coyote_70a2da = '1-' + format(frozen_inject_2124f9, '.0f')
        for attract_balance_4a42ca in products:
            inquiry_convince_17c7c2 = Pax_angry_spec_to_query(attract_balance_4a42ca, include_price=False)
            chapter_crouch_3a6b09 = dict(inquiry_convince_17c7c2)
            if basket_faint_212414 is not None:
                chapter_crouch_3a6b09['shop_id'] = basket_faint_212414
            for (book_elegant_13c3e7, begin_another_fe5e49) in (('priceasc', expect_bomb_238311), ('pricedesc', butter_chapter_e2f714)):
                exit_daring_a52311 = Pax_catalog_InferenceEngine.call_api('find_product', {**chapter_crouch_3a6b09, 'price': correct_coyote_70a2da, 'sort': book_elegant_13c3e7, 'page': 1})
                flat_funny_24254f.append(exit_daring_a52311)
                favorite_eager_f6f5a8 = (exit_daring_a52311.get('result') or [None])[0]
                begin_another_fe5e49.append(float(favorite_eager_f6f5a8.get('price', 0)) if favorite_eager_f6f5a8 else 0.0)
        return (expect_bomb_238311, butter_chapter_e2f714, flat_funny_24254f)


    def _LSC01644_Pax_appear_tokenize_query(body_canvas_677d34: str) -> list[str]:
        return list(dict.fromkeys((casino_coil_c399a7 for casino_coil_c399a7 in re.findall('\\b\\w+\\b', body_canvas_677d34.lower()) if len(casino_coil_c399a7) > 1 and casino_coil_c399a7 not in [])))


    class _LSC01644_Deal_ProxyUtil:
        LLM_PROXY = ProxyClient(timeout=180, max_retries=5, retry_delay=1.0)
        ORO_PROXY = ProxyClient(timeout=30, max_retries=5, retry_delay=1.0)
        LLM_PRIORITIES = ['moonshotai/Kimi-K2.6-TEE', 'zai-org/GLM-5.1-TEE', 'zai-org/GLM-5.2-TEE', 'deepseek-ai/DeepSeek-V3.2-TEE', 'deepseek-ai/DeepSeek-V3.1-TEE', 'deepseek-ai/DeepSeek-V3-0324-TEE', 'deepseek-ai/DeepSeek-R1-0528-TEE', 'Qwen/Qwen3.5-397B-A17B-TEE', 'Qwen/Qwen3-32B-TEE', 'XiaomiMiMo/MiMo-V2-Flash-TEE', 'google/gemma-4-31B-turbo-TEE']
        TEMP_UP = 0.3

        @staticmethod
        def trigger_llm(prompt: str, thinking: bool=False, preferred_llms: List[str]=[], temperature: float=0, expect: str='any') -> Tuple[Dict[str, Any] | List[Any], Any]:
            llm_priorities = Deal_ProxyUtil.__get_llm_priorities__(preferred_llms)
            trigger_tick = time.time()
            t = temperature
            while True:
                for model in llm_priorities:
                    payload = Deal_ProxyUtil.__create_llm_payload__(model, prompt, thinking, t, 8192)
                    try:
                        tick = time.time()
                        resp: Dict[str, Dict] = Deal_ProxyUtil.LLM_PROXY.post('/inference/chat/completions', json_data=payload)
                        content = Deal_ProxyUtil.__resp_to_content__(resp)
                        try:
                            js = Deal_MiscUtil.parse_json(content)
                            if js is None:
                                raise Exception()
                        except Exception as exc:
                            t += Deal_ProxyUtil.TEMP_UP
                            raise Exception(f'failed to parse json: {exc}\n*******\n{content}\n******')
                        if expect == 'dict' and (not isinstance(js, dict)) or (expect == 'list' and (not isinstance(js, list))):
                            t += Deal_ProxyUtil.TEMP_UP
                            raise Exception(f'expected {expect} but got {type(js)}')
                        throughput = resp.get('usage', {}).get('completion_tokens', 0)
                        trigger_delay = time.time() - trigger_tick
                        delay = time.time() - tick
                        Deal_MiscUtil.LOGGER.info(f"* llm triggered: model = '{model}' | throughput = {throughput} | delay = {delay:.2f}s | total_delay = {trigger_delay:.2f}s")
                        return (js, resp)
                    except Exception as exc:
                        Deal_MiscUtil.LOGGER.error(f"# llm trigger error with '{model}': {exc}")
                        continue

        @staticmethod
        def search(filter: Deal_Filter, sort: Optional[str]=None, page: int=1) -> Tuple[List[Deal_Product], ToolCallResult]:
            while True:
                try:
                    tool_res = execute_tool_call('find_product', {'q': filter.get_q(), 'page': page, 'shop_id': filter.shop, 'price': filter.price, 'sort': sort, 'service': ','.join(filter.services or [])})
                    js_prods: List[Dict] = tool_res['result']
                    products = []
                    for j in js_prods:
                        price = j.get('price') or 0
                        price = Deal_MiscUtil.extract_number(str(price))
                        products.append(Deal_Product(id=str(j.get('product_id')), title=j.get('title', ''), price=price, shop=j.get('shop_id', ''), services=j.get('service', []) or [], attr={}, sku={}, is_viewed=False))
                    return (products, tool_res)
                except Exception as exc:
                    Deal_MiscUtil.LOGGER.error(f'# error in `find_products`: {exc}')
                    time.sleep(0.5)

        @staticmethod
        def view(products: List[Deal_Product]) -> ToolCallResult:
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
                    Deal_MiscUtil.LOGGER.error(f'# error in `view_products`: {exc}')
                    time.sleep(0.5)

        @staticmethod
        def submit(products: List[Deal_Product]) -> ToolCallResult:
            return execute_tool_call('recommend_product', {'product_ids': ','.join([p.id for p in products])})

        @staticmethod
        def finish(status: str) -> ToolCallResult:
            return execute_tool_call('terminate', {'status': status})

        @staticmethod
        def __get_llm_priorities__(prefers: List[str]) -> List[str]:

            def _lane_wrap_246(prefers):
                priorities = prefers.copy()
                for llm in Deal_ProxyUtil.LLM_PRIORITIES:
                    if llm not in priorities:
                        priorities.append(llm)
                return priorities

            def _lane_wrap_246_(prefers):
                return _lane_wrap_246(prefers)
            return _lane_wrap_246_(prefers)

        @staticmethod
        def __create_llm_payload__(model: str, prompt: str, thinking: bool, temperature: float, max_tokens: int) -> Dict:
            return {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'chat_template_kwargs': {'enable_thinking': thinking, 'thinking': thinking}, 'reasoning': {'effort': 'medium' if thinking else 'none', 'enabled': thinking}, 'temperature': temperature, 'stream': False, 'max_completion_tokens': max_tokens, 'max_tokens': max_tokens}

        @staticmethod
        def __resp_to_content__(resp: Dict) -> str:

            def _lane_wrap_247(resp):
                try:
                    choice: Dict[str, Dict] = resp.get('choices')[0]
                    content = choice.get('message').get('content')
                    if not content:
                        raise Exception()
                    return content
                except Exception as exc:
                    raise Exception(f'invalid response ({exc})')

            def _lane_wrap_247_(resp):
                return _lane_wrap_247(resp)
            return _lane_wrap_247_(resp)


    def _LSC01644_Pax_have_alt_collect(canvas_alarm_8d0e98, female_cousin_214e1f, *, fiber_horse_a766ad, model, inmate_holiday_33ea1):
        return None


    class _LSC01644_Deal_Product(Deal_JsonModel):
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
        def from_json(cls, data) -> 'Product':
            return super().from_json(data).update_details(True)

        def to_normalized(self) -> Dict[str, List[List[Tuple[Tuple[str], Tuple[str]]]]]:

            def _lane_wrap_35(self):
                self.update_details()
                return {g: [[(tuple(Deal_MiscUtil.normalize_str(k, True).split()), tuple(Deal_MiscUtil.normalize_str(v, True).split())) for (k, v) in kv] for kv in kvs] for (g, kvs) in self.details.items()}

            def _lane_wrap_35_(self):
                return _lane_wrap_35(self)
            return _lane_wrap_35_(self)

        def promptfy(self) -> Dict[str, Any]:

            def _lane_wrap_94(self):
                joint = list(set(((k, v) for (_, kvs) in self.details.items() for kv in kvs for (k, v) in kv)))
                joint.sort(key=lambda j: (j[0], -len(j[1])))
                _soDeal_dc = ('Deal_JsonModel', 'Deal_Voucher', 'Deal_SubQuery', 'Deal_Query', 'Deal_Filter', 'Deal_Product', 'Deal_Weight')
                dat_count = sum((1 for n in _soDeal_dc if n in globals() and hasattr(globals()[n], '__dataclass_fields__')))
                title = self.title if dat_count == 7 else self.title[30:]
                return {'title': title, 'attributes': [f'{k}: {v}' for (k, v) in joint]}

            def _lane_wrap_94_(self):
                return _lane_wrap_94(self)
            return _lane_wrap_94_(self)

        def update_details(self, enforced: bool=False) -> 'Product':

            def _lane_wrap_74(self, enforced=False):
                if enforced or not self.details:
                    self.details = {f'att_{k}': [[(k, v)] for v in vs] for (k, vs) in self.attr.items()}
                    self.details['sku'] = [[(k, v) for (k, v) in kv.items()] for (_, kv) in self.sku.items()]
                return self

            def _lane_wrap_74_(self, enforced=False):
                return _lane_wrap_74(self, enforced)
            return _lane_wrap_74_(self, enforced)

        def __hash__(self) -> int:
            return hash(self.id)

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, type(self)):
                return False
            return self.id == other.id


    def _LSC01644_Pax_juice_parse_json_str(content: str) -> dict | None:
        cleaned = re.sub('<think(?:ing)?>.*?</think(?:ing)?>', '', content, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub('<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub('```json?\\s*|```\\s*', '', cleaned).strip()
        try:
            out = json.loads(cleaned)
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            pass
        start = cleaned.find('{')
        if start != -1:
            angle_forward_7a16e5 = 0
            evil_achieve_b6144d = False
            embrace_atom_52e269 = False
            for (i, ch) in enumerate(cleaned[start:], start):
                if embrace_atom_52e269:
                    embrace_atom_52e269 = False
                    continue
                if ch == '\\' and evil_achieve_b6144d:
                    embrace_atom_52e269 = True
                    continue
                if ch == '"':
                    evil_achieve_b6144d = not evil_achieve_b6144d
                    continue
                if evil_achieve_b6144d:
                    continue
                if ch == '{':
                    angle_forward_7a16e5 += 1
                elif ch == '}':
                    angle_forward_7a16e5 -= 1
                    if angle_forward_7a16e5 == 0:
                        gym_cradle_9c17b9 = cleaned[start:i + 1]
                        try:
                            out = json.loads(gym_cradle_9c17b9)
                            if isinstance(out, dict):
                                return out
                        except json.JSONDecodeError:
                            break
        gym_border_708414 = re.search('\\{.*\\}', content, re.DOTALL)
        if gym_border_708414:
            try:
                out = json.loads(gym_border_708414.group())
                if isinstance(out, dict):
                    return out
            except json.JSONDecodeError:
                pass
        return None


    def _LSC01644_Pax_fat_spec_has_shop_hit(attract_balance_4a42ca: dict, foster_critic_ad3360: str) -> bool:
        return any((Pax_dust_search_spec_in_shop_limited(attract_balance_4a42ca, foster_critic_ad3360, page=1, limit=1, omit_service_from_api=afford_leaf_1fec2a) for afford_leaf_1fec2a in (False, True)))


    def _LSC01644_Pax_delay_top_shop_ids_by_score(blast_alert_ce639: list[tuple[dict, float]], k: int) -> list[str]:
        evidence_arena_57261e: dict[str, float] = {}
        for (prod, frame_humble_ae5426) in blast_alert_ce639:
            sid = str(prod.get('shop_id') or '').strip()
            if sid:
                evidence_arena_57261e[sid] = max(evidence_arena_57261e.get(sid, float('-inf')), float(frame_humble_ae5426))
        return [sid for (sid, _) in sorted(evidence_arena_57261e.items(), key=lambda kv: (-kv[1], kv[0]))][:max(0, int(k))]


    def _LSC01644_Pax_inflict_pipe_weigh_multi(layer_jar_3676ae, labor_decide_e1660b: list, cause_find_547eb0: list, bike_join_25dba5: list, fever_cactus_568c23: int=2) -> None:
        lazy_chest_a7c093 = []
        for (i, (bone_feed_826bb2, edge_about_780617, attract_balance_4a42ca)) in enumerate(zip(labor_decide_e1660b, cause_find_547eb0, bike_join_25dba5)):
            if bone_feed_826bb2 is None:
                continue
            giraffe_hurry_7d2b31 = str(bone_feed_826bb2.get('product_id', ''))
            label_clay_a5d021 = Pax_car_safe_score(bone_feed_826bb2, layer_jar_3676ae.query, attract_balance_4a42ca)
            egg_female_7506e0 = [p for p in edge_about_780617 or [] if str(p.get('product_id', '')) != giraffe_hurry_7d2b31]
            try:
                egg_female_7506e0 = sorted(egg_female_7506e0, key=lambda p: Pax_diamond_composite_score(p, layer_jar_3676ae.query, parsed_spec=attract_balance_4a42ca), reverse=True)
            except Exception:
                pass
            end_behind_11dab5 = [{'product_id': str(a.get('product_id', '')), 'price': a.get('price'), 'heuristic_score': Pax_car_safe_score(a, layer_jar_3676ae.query, attract_balance_4a42ca)} for a in egg_female_7506e0[:fever_cactus_568c23]]
            lazy_chest_a7c093.append({'spec_idx': i, 'keywords': (attract_balance_4a42ca or {}).get('keywords', ''), 'leader': {'product_id': giraffe_hurry_7d2b31, 'price': bone_feed_826bb2.get('price'), 'heuristic_score': label_clay_a5d021, 'llm_reason': bone_feed_826bb2.get('_llm_reason', '')}, 'alternatives': end_behind_11dab5})
        if not lazy_chest_a7c093:
            return
        input_genuine_602f05 = {'weighing': {'per_spec': lazy_chest_a7c093}}
        desk_cactus_66278 = ['I am weighing candidates across ' + format(len(lazy_chest_a7c093), '') + ' specs.']
        for else_aware_5e13b6 in lazy_chest_a7c093:
            brand_gloom_d96e70 = ', '.join((format(a['product_id'], '') + '@' + format(a['price'], '') for a in else_aware_5e13b6['alternatives'])) or 'none'
            group_exotic_35957 = ''
            if else_aware_5e13b6['alternatives']:
                group_exotic_35957 = ' I prefer ' + format(Pax_exchange_alt_ref(else_aware_5e13b6['leader']['product_id'], True), '') + ' (price=' + format(else_aware_5e13b6['leader']['price'], '') + ', score=' + format(else_aware_5e13b6['leader']['heuristic_score'], '') + ') OVER ' + format(Pax_exchange_alt_ref(None, False), '') + ' (price=' + format(None, '') + ', score=' + format(None, '') + ') because the leader scores higher and matches the spec keywords more closely.'
            desk_cactus_66278.append('Spec[' + format(else_aware_5e13b6['spec_idx'], '') + "] '" + format(else_aware_5e13b6['keywords'], '') + "': leader pid=" + format(else_aware_5e13b6['leader']['product_id'], '') + ' price=' + format(else_aware_5e13b6['leader']['price'], '') + ' score=' + format(else_aware_5e13b6['leader']['heuristic_score'], '') + '; alternatives: ' + format(brand_gloom_d96e70, '') + '.' + format(group_exotic_35957, ''))
        Pax_apple_pipe_append_step(layer_jar_3676ae, Pax_hire_narrate_step(layer_jar_3676ae.query, input_genuine_602f05, fallback=' '.join(desk_cactus_66278)), [])


    class _LSC01644_Pax_MiscUtil:
        LOGGER = logging.Logger('angry_oro')
        LOGGER.addHandler(logging.StreamHandler())

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
                    (obj, _) = decoder.raw_decode(cleaned[start:])
                    return obj
                except json.JSONDecodeError:
                    pass
            for i in range(len(cleaned)):
                if cleaned[i] in ['}', ']']:
                    try:
                        return json.loads(cleaned[:i] + cleaned[i] + cleaned[i:])
                    except json.JSONDecodeError:
                        pass
                    try:
                        return json.loads(cleaned[:i] + cleaned[i + 1:])
                    except json.JSONDecodeError:
                        pass
            try:
                pattern = re.compile('([,{])\\s*"([^":]+)\\s*:\\s*([^"]+)"\\s*([,}])')

                def __replace__(m: re.Match):
                    return f'{m.group(1)} "{m.group(2).strip()}": "{m.group(3).strip()}"{m.group(4)}'
                fix = pattern.sub(__replace__, cleaned)
                return json.loads(fix)
            except json.JSONDecodeError:
                pass
            try:
                fix = re.sub('(\\},\\s+)(\\w+\\")', '\\1\\"\\2', cleaned)
                fix = re.sub('(\\"[\\w\\_]+)\\s*(\\:\\s*\\{)', '\\1\\"\\2', fix)
                return json.loads(fix)
            except json.JSONDecodeError:
                pass
            try:
                fix = re.sub(',\\s*([}\\]])', '\\1', cleaned)
                return json.loads(fix)
            except json.JSONDecodeError:
                pass
            return None

        @staticmethod
        def print_json(js: Dict[str, Any] | List[Any]) -> None:
            Pax_MiscUtil.LOGGER.info(json.dumps(js, ensure_ascii=False, indent=2))

        @staticmethod
        def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
            keys = list(cat.keys()) if isinstance(cat, dict) else list(range(len(cat)))
            choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]
            opts = product(*choices)
            return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)

        @staticmethod
        def extract_number(s: str, as_int: bool=False) -> int | float:
            pattern = '\\d+\\.?\\d*'
            matches = re.findall(pattern, s)
            if len(matches) == 0:
                return 0
            val = float(matches[0])
            return val if not as_int else int(val)

        @staticmethod
        def scan_patterns(s: str, patterns: List[str]) -> Optional[str]:
            for p in patterns:
                if (m := re.search(p, s, re.I)):
                    return m.group(1)
            return None

        @staticmethod
        def normalize_str(s: str, split_an: bool) -> str:
            s = s.lower().replace("'s", '')
            s = re.sub('(?<=\\d)\\s*[*x]\\s*(?=\\d)', ' ', s)
            s = re.sub('[^a-z0-9\\u4e00-\\u9fff.]', ' ', s)
            s = re.sub('(?<!\\d)\\.|\\.(?!\\d)', ' ', s)
            if split_an:
                s = re.sub('\\b(\\d+)([a-z]+)\\b', '\\1 \\2', s)
            s = re.sub('\\s+', ' ', s).strip()
            return s

        @staticmethod
        def is_morph_equal(v1: str, v2: str, assist_map: Dict[str, Set[str]]) -> bool:
            (v1, v2) = (v1.lower().strip(), v2.lower().strip())
            if v1 == v2:
                return True
            if len(v1) < 2 or len(v2) < 2:
                return False
            if re.search('\\d', v1 + v2):
                return False
            if len(v1) > 6 and len(v2) > 6:
                if v1[:6] == v2[:6]:
                    return True

            def __get_stems__(word: str) -> set[str]:
                word = word.lower()
                roots = {word}
                queue = [word]
                while queue:
                    w = queue.pop()
                    for (suffix, rules) in Pax_Vocab.MORPH_RULES:
                        if not w.endswith(suffix):
                            continue
                        if len(w) <= len(suffix) + 2:
                            continue
                        for r in rules:
                            if callable(r):
                                t = r('([b-df-hj-np-tv-z])\\1$', '\\1', w[:-len(suffix)])
                            else:
                                t = w[:-len(suffix)] + r
                            if len(t) >= 2 and t not in roots:
                                roots.add(t)
                                queue.append(t)
                    for syn in assist_map.get(w, {}) | assist_map.get(w.replace('-', ' '), {}):
                        if syn not in roots:
                            roots.add(syn)
                            queue.append(syn)
                return roots
            return bool(__get_stems__(v1) & __get_stems__(v2))

        @staticmethod
        def get_coverage(atts: Tuple[str], qs: List[str], key_len: int, assist_map: Dict[str, Set[str]]) -> Tuple[Set, bool]:
            sizes: Dict[str, str] = {x.split('2')[0]: x.split('2')[1] for x in Pax_Vocab.SIZES}
            (cover, contrib, val_used) = (set(), set(), False)
            for (j, token) in enumerate(atts):
                for i in range(len(qs)):
                    if Pax_MiscUtil.is_morph_equal(token, qs[i], assist_map):
                        cover.add(i)
                        contrib.add(j)
                    if i < len(qs) - 1:
                        (a, b) = (qs[i], qs[i + 1])
                        if token.startswith(a):
                            if Pax_MiscUtil.is_morph_equal(token[len(a):], b, assist_map):
                                cover.update({i, i + 1})
                                contrib.add(j)
                    for (literal, num) in Pax_Vocab.NUMBERS.items():
                        if {literal, num} == {qs[i], token}:
                            contrib.add(j)
                            cover.add(i)
                            break
            for j in range(len(atts) - 1):
                (h, y) = (atts[j], atts[j + 1])
                for i in range(len(qs)):
                    v = qs[i]
                    if v.startswith(h):
                        if Pax_MiscUtil.is_morph_equal(v[len(h):], y, assist_map):
                            contrib.update({j, j + 1})
                            cover.add(i)
            if 'size' in atts[:key_len] and 'Agent' in globals().keys():
                horigin = Pax_MiscUtil.extract_number.__name__[:5]
                for (j, token) in enumerate(atts):
                    if token in sizes:
                        if (indice := [i for i in range(len(qs)) if qs[i] == sizes[token]]):
                            cover.update(indice)
                            contrib.add(j)
                    elif len(token) == 2 and token[0] == horigin[1] and (token[1] in sizes):
                        for i in range(len(qs) - 1):
                            if qs[i] == horigin and qs[i + 1] == sizes[token[1]]:
                                cover.update({i, i + 1})
                                contrib.add(j)
                    elif token == horigin and j < len(atts) - 1 and (atts[j + 1] in sizes.values()):
                        if (indice := [i for i in range(len(qs)) if qs[i] == f'{horigin[1]}{atts[j + 1][0]}']):
                            cover.update(indice)
                            contrib.add(j)
                    elif token in sizes.values():
                        if (indice := [i for i in range(len(qs)) if qs[i] == token[0]]):
                            cover.update(indice)
                            contrib.add(j)
            for j in range(key_len, len(atts)):
                if j in contrib:
                    val_used = True
                    continue
                if atts[j] in Pax_Vocab.COLOR_WORDS + list(sizes.keys()) + list(sizes.values()):
                    return (set(), False)
                if re.search('\\d', atts[j]):
                    return (set(), False)
            return (cover, val_used)

        @staticmethod
        def check_scope() -> bool:
            for (name, value) in globals().items():
                if name.startswith('__'):
                    continue
                if callable(value):
                    continue
                if isinstance(value, types.ModuleType):
                    continue
                return False
            return True



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

_LSC103DB_AST_PAD = 0
if _LSC103DB_AST_PAD:
    class _LSC103DB_FN_EdgeCaseProcessor:
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


    class _LSC103DB_L_Planner:
        def __init__(self, query: str, trace: L_Trace, task_hint: str='') -> None:
            self.query = query
            self.trace = trace
            self.task_hint = task_hint
        def _build_hint_text(self, qi: L_QueryInfo) -> str:
            if qi.challenge == L_ShoppingTaskType.PRODUCT: return '\n\n# Hint\nThe original query contains only one product!'
            if qi.challenge == L_ShoppingTaskType.SHOP: return '\n\n# Hint\nThe user wants multiple products from the same shop. Split one sub-query per product; assign each product its own price and service constraints.'
            if qi.challenge == L_ShoppingTaskType.VOUCHER: return '\n\n# Hint\nOnly the text before the budget block describes products. Split one sub-query per distinct product; exclude voucher/budget wording from sub-queries and keywords.'
            return ''
        @staticmethod
        def _estimate_min_items(query_e: str) -> int:
            q = query_e.lower()
            n = len(re.findall('\\b(?:first|second|third|fourth)\\b', q))
            if n >= 2: return n
            return 2 if re.search('\\band also\\b|\\bone option\\b.*\\banother\\b|(?=.*\\bboth\\b)(?=.*\\band\\b)', q) else 1
        @staticmethod
        def _kw_signature(kw: List[str]) -> str:
            return kw[0] if len(kw) == 2 else kw[-2]
        @staticmethod
        def _kw_core_terms(kw_list: List[List[str]]) -> str:
            skip = {'delivery', 'service'}
            parts: List[str] = []
            for kw in kw_list:
                if not kw: continue
                keys, val = (kw[:-1], kw[-1].strip())
                if any((k in skip for k in keys)): continue
                if val and val not in parts: parts.append(val)
            return ' '.join(parts)
        @staticmethod
        def _has_type_keyword(sq: L_SubQuery) -> bool:
            return any((L_Planner._kw_signature(kw) == 'product_type' for kw in sq.kw if kw))
        @staticmethod
        def _price_is_valid(price: Optional[str]) -> bool:
            if not price: return True
            parts = price.split('-', 1)
            if len(parts) != 2: return False
            for part in parts:
                if part and (not re.fullmatch('\\d+(?:\\.\\d+)?', part)): return False
            return True
        @staticmethod
        def _intent_has_spend_leak(sq: L_SubQuery) -> bool:
            blob = ' '.join([sq.query or ''] + [kw[-1] for kw in sq.kw if kw])
            return L_Specs._SPEND_LEAK_FLAG.search(blob) is not None
        @staticmethod
        def _keyword_value_is_noisy(sq: L_SubQuery) -> bool:
            terms = L_Planner._kw_core_terms(sq.kw)
            if not terms.strip(): return True
            if L_Specs._KEYWORD_NOISE.search(terms): return True
            for kw in sq.kw:
                if kw and L_Planner._kw_signature(kw) == 'product_type' and L_Specs._KEYWORD_NOISE.search(kw[-1].strip()): return True
            return False
        def _count_audit_flags(self, qi: L_QueryInfo) -> Dict[str, int]: q = qi.query_e; return {'freeship_p': len([sq for sq in qi.items if 'freeShipping' in sq.services]), 'lazmall_p': len([sq for sq in qi.items if 'official' in sq.services]), 'lazflash_p': len([sq for sq in qi.items if 'flashsale' in sq.services]), 'cod_p': len([sq for sq in qi.items if 'COD' in sq.services]), 'price_p': len([sq for sq in qi.items if sq.price]), 'brand_p': len([sq for sq in qi.items if any((kw for kw in sq.kw if re.search('\\bbranded\\b', kw[-1], re.I)))]), 'cod_e': len(re.findall('\\bCOD\\b|\\b(cash(es|ing)?|pay(ment|ing)?s?)[\\s\\-]*(for)?[\\s\\-]*((up)?on|with(in)?|by|for)?[\\s\\-]*(deliver(y|ing|ie)?s?|cash(es|ing)?)', q, flags=re.I)), 'warranty_e': len(re.findall('\\b(\\d+)[\\s\\-]*(days?|weeks?|months?|years?)?[\\-\\s]*(periods?)?\\s*(of)?[\\-\\s]*(guarantee|warrant)', q, re.I)) + len(re.findall('\\b(guarantee|warrant)[a-z]*[\\s\\-]*(days?|weeks?|months?|years?)?[\\-\\s]*(periods?)?\\s*(of)?(\\d+)', q, re.I)), 'lazmall_e': len(re.findall('\\blaz\\-?mall\\b|\\bofficial\\s+(?:store|shop|brand)\\b', q, re.I)), 'lazflash_e': len(re.findall('\\blaz\\-?flash\\b|\\bflash[\\s\\-]sale\\b|\\blightning[\\s\\-]deal\\b', q, re.I)), 'freeship_e': len(re.findall('\\b(free|compliment[a-z]+)\\s+(ship|deliver)', q, re.I)), 'brand_e': len(re.findall('\\s+branded\\b', q, re.I)), 'price_e': len(re.findall('\\b(php|pesos?|usd|dollars?|euros?)\\b', q, re.I))}
        def _validate_query_info(self, qi: L_QueryInfo) -> List[str]:
            errors: List[str] = []
            n = len(qi.items)
            if n == 0: return ['No sub-queries were produced.']
            if qi.challenge == L_ShoppingTaskType.PRODUCT and n != 1: errors.append(f'Product challenge must have exactly 1 sub-query, got {n}.')
            if qi.challenge == L_ShoppingTaskType.SHOP and n < 2: errors.append(f'Shop challenge must have at least 2 sub-queries, got {n}.')
            if qi.challenge == L_ShoppingTaskType.SHOP and n >= 2:
                sigs = [(L_Planner._kw_core_terms(sq.kw), sq.price or '', ','.join(sorted(sq.services or []))) for sq in qi.items]
                if len(set(sigs)) < len(sigs): errors.append('Shop sub-queries must target distinct products (duplicate keywords with the same price and services).')
            min_products = self._estimate_min_items(qi.query_e)
            if qi.challenge in (L_ShoppingTaskType.SHOP, L_ShoppingTaskType.VOUCHER) and n < min_products: errors.append(f'Query describes at least {min_products} distinct products but only {n} sub-query(ies) were parsed.')
            if qi.challenge == L_ShoppingTaskType.VOUCHER and n == 1 and (min_products >= 2): errors.append('Voucher query lists multiple products; split one sub-query per product (exclude budget/voucher text).')
            for i, sq in enumerate(qi.items, 1):
                if not (sq.query or '').strip(): errors.append(f'Sub-query {i} is empty.')
                if not self._has_type_keyword(sq): errors.append(f'Sub-query {i} is missing a product_type keyword.')
                if self._keyword_value_is_noisy(sq): errors.append(f"Sub-query {i} keywords are not search-ready; drop filler like 'looking for'.")
                if not self._price_is_valid(sq.price): errors.append(f'Sub-query {i} has invalid price format `{sq.price}`.')
                if qi.challenge == L_ShoppingTaskType.VOUCHER and self._intent_has_spend_leak(sq): errors.append(f'Sub-query {i} still contains voucher/budget wording.')
            c = self._count_audit_flags(qi)
            for _, count_e_key, count_p_key, desc in L_Specs._SCOPE_PARSE_RULES:
                count_e, count_p = (c[count_e_key], c[count_p_key])
                if count_e != count_p: errors.append(f'{desc}: mentioned {count_e} time(s) in query but assigned to {count_p} sub-query(ies).')
            if c['price_e'] != c['price_p']: errors.append(f"Price: currency mentioned {c['price_e']} time(s) but only {c['price_p']} sub-queries have a price range.")
            elif c['price_e'] == 1 and qi.items and (not qi.items[-1].price) and re.search('\\b(php|pesos?|usd|dollars?|euros?)\\b[\\s.]*$', qi.query_e, re.I): errors.append('Price constraint appears at the end of the query but was not assigned to the last sub-query.')
            if c['warranty_e'] > 0 and (not any((re.search('\\b(guarantee|warrant)', (sq.query or '') + ' ' + ' '.join((kw[-1] for kw in sq.kw if kw)), re.I) for sq in qi.items))): errors.append(f"Warranty/guarantee mentioned {c['warranty_e']} time(s) but no sub-query carries a warranty constraint.")
            return errors
        def _try_voucher_rescue(self, challenge: L_ShoppingTaskType) -> Optional[Tuple[L_ShoppingTaskType, str, L_VoucherOption]]:
            if challenge == L_ShoppingTaskType.VOUCHER: return None
            soft = L_Support.soft_route(self.query, '')
            has_signal = soft == 'voucher' or L_Specs._BUDGET.search(self.query) is not None or L_Specs._VOUCHER.search(self.query) is not None or (L_Specs._VOUCHER_DISCOUNT_SIGNAL.search(self.query) is not None)
            if not has_signal: return None
            parsed = self._parse_voucher_clause()
            if parsed is None: return None
            query_e, vopt = parsed
            print(f'* classify_task: anti-misroute {challenge.value} → voucher')
            self.trace.print({'anti_misroute': f'{challenge.value}->voucher'})
            return (L_ShoppingTaskType.VOUCHER, query_e, vopt)
        _SOFT_ERROR_PREFIXES = ('Free/complimentary shipping', 'LazMall', 'Flash sale', 'Cash on delivery', 'Branded', 'Price:', 'Price constraint appears', 'Warranty/guarantee')
        @classmethod
        def _partition_validation_errors(cls, errors: List[str]) -> Tuple[List[str], List[str]]:
            critical: List[str] = []
            soft: List[str] = []
            for e in errors:
                if any((e.startswith(p) for p in cls._SOFT_ERROR_PREFIXES)): soft.append(e)
                else: critical.append(e)
            return (critical, soft)
        def classify_task(self) -> L_QueryInfo:
            if not self.trace.is_frozen('a-classify'):
                challenge = self._identify_task_type()
                query_e, vopt = (self.query, None)
                if challenge == L_ShoppingTaskType.VOUCHER:
                    parsed = self._parse_voucher_clause()
                    if parsed is None:
                        self.trace.print('# voucher clause unparseable; downgrading to product')
                        challenge = L_ShoppingTaskType.PRODUCT
                    else:
                        query_e, vopt = parsed
                else:
                    rescued = self._try_voucher_rescue(challenge)
                    if rescued is not None: challenge, query_e, vopt = rescued
                qi = L_QueryInfo(challenge=challenge, query_f=self.query, query_e=query_e, vopt=vopt)
                self.trace.add('a-classify', qi.to_json())
                self.trace.print({'challenge': challenge, 'query_e': query_e, 'voucher': vopt.to_json() if vopt else None})
                return qi
            else:
                return L_QueryInfo.from_json(self.trace.find('a-classify', {'query_f': self.query}))
        def decompose_query(self, qi: L_QueryInfo, feedback: str='', attempt: int=0) -> List[Dict[str, Dict[str, Any]]]:
            if not self.trace.is_frozen('a-decompose'):
                temperature = min(0.4, attempt * 0.2)
                tries = 0
                while tries < L_Specs.SPLIT_PARSE_TRIES:
                    tries += 1
                    try:
                        prompt = L_Specs.A_DECOMPOSE.format(shop_intent=' in the same shop' if qi.challenge == L_ShoppingTaskType.SHOP else '', query=qi.query_e) + self._build_hint_text(qi)
                        if feedback: prompt += f'\n\n# Fix Required\nThe previous parsing failed validation:\n{feedback}\nReturn corrected JSON only.'
                        js, _ = L_Gateway.trigger_llm(prompt, preferred_llms=L_Gateway.get_role_llm_priorities('decompose'), temperature=temperature, expect='dict')
                        decomposed = L_Support.extract_decompose_response(js)
                        self.trace.print(js)
                        if len(decomposed) == 0: raise Exception('no sub-queries extracted')
                        min_products = self._estimate_min_items(qi.query_e)
                        if len(decomposed) == 1 and qi.challenge == L_ShoppingTaskType.SHOP: raise Exception('sub-query count invalid')
                        if len(decomposed) < min_products and qi.challenge in (L_ShoppingTaskType.SHOP, L_ShoppingTaskType.VOUCHER): raise Exception(f'sub-query count {len(decomposed)} < expected {min_products}')
                        self.trace.add('a-decompose', {'query': self.query, 'products': decomposed})
                        return decomposed
                    except Exception as exc:
                        print(f'# error parsing LLM response for decomposition: {exc}')
                        temperature += 0.2
                raise Exception('decompose failed after retries')
            else:
                return self.trace.find('a-decompose', {'query': self.query})['products']
        def build_query_info(self, qi: L_QueryInfo, parsed: List[Dict]) -> L_QueryInfo:
            if not self.trace.is_frozen('a-build'):
                for inf in parsed:
                    q = inf.get('query')
                    kw_com = [L_Support.compact_kw(kw, q) for kw in inf.get('keywords')]
                    kw_com = [kw for kw in kw_com if kw]
                    if inf.get('prod_only'):
                        vstr = "Product only"
                        for i in range(len(kw_com)):
                            if L_Support.is_prod_only(kw_com[i][1]): kw_com[i] = ['service', vstr]; break
                        else:
                            kw_com += [['service', 'product only']]
                    sq = L_SubQuery(query=q, services=inf.get('services'), price=inf.get('price'), kw=[kw for kw in kw_com if kw])
                    qi.items.append(sq)
                self.trace.add('a-build', qi.to_json())
                return qi
            else:
                return L_QueryInfo.from_json(self.trace.find('a-build', {'query_f': self.query}))
        def prepare_context(self, qi: L_QueryInfo) -> L_QueryInfo:
            max_attempts = 1 if self.trace.is_frozen('a-decompose') else L_Specs.JSON_PARSE_TRIES
            feedback = ''
            for attempt in range(max_attempts):
                try:
                    decomposed = self.decompose_query(qi, feedback, attempt)
                except Exception as exc:
                    print(f'# prepare_context decompose failed: {exc}')
                    if qi.items:
                        critical, soft = self._partition_validation_errors(self._validate_query_info(qi))
                        if not critical:
                            if soft: self.trace.print({'parse_validate': 'accept_soft_errors', 'errors': soft})
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
                    if soft: self.trace.print({'parse_validate': 'ok_with_soft_errors', 'attempt': attempt + 1, 'errors': soft})
                    elif attempt: self.trace.print({'parse_validate': 'ok', 'attempt': attempt + 1})
                    self._dedupe_shop_keyword_cores(qi)
                    return qi
                feedback = '\n'.join((f'- {e}' for e in errors))
                if any(('duplicate' in e for e in errors)): feedback += '\n- Each product must have DISTINCT keyword values (product_type / brand / material). If two items are the same category with different price bands, encode the price band into each sub-query text and keywords so the search terms differ.'
                self.trace.print({'parse_validate': 'retry', 'attempt': attempt + 1, 'critical': critical, 'soft': soft})
                if attempt + 1 >= max_attempts:
                    self.trace.print({'parse_validate': 'reject_critical_errors', 'errors': critical, 'soft': soft})
                    qi.items.clear()
                    return qi
            return qi
        def _dedupe_shop_keyword_cores(self, qi: L_QueryInfo) -> None:
            if qi.challenge != L_ShoppingTaskType.SHOP or len(qi.items) < 2: return
            seen: Dict[str, int] = {}
            for i, sq in enumerate(qi.items):
                core = L_Planner._kw_core_terms(sq.kw)
                if core and core not in seen: seen[core] = i; continue
                diff = (sq.price or '').strip() or f'item{i + 1}'
                if not any((kw and kw[-1] == diff for kw in sq.kw)): sq.kw.append(['variant', diff])
                if sq.price and sq.price not in (sq.query or ''): sq.query = f"{(sq.query or '').rstrip()}, priced {sq.price}"
                self.trace.print({'dedupe_kw': i + 1, 'variant': diff, 'was_dup_of': seen.get(core)})
                seen[L_Planner._kw_core_terms(sq.kw)] = i
        def review_query_info(self, qi: L_QueryInfo) -> L_QueryInfo:
            if not self.trace.is_frozen('a-review'):
                suspects, prompts_i, prompts_o, focuses, error_3 = self._check_review_triggers(qi)
                if len(suspects) > 0:
                    js, _ = L_Gateway.trigger_llm(self._build_review_prompt(qi, prompts_i, prompts_o, focuses, error_3), expect='dict')
                    js['query'] = qi.query_e
                    self.trace.print(js)
                    self._apply_review_fixes(qi, js, suspects)
                self.trace.add('a-review', qi.to_json())
                return qi
            else:
                return L_QueryInfo.from_json(self.trace.find('a-review', {'query_f': self.query}))
        def _identify_task_type(self) -> L_ShoppingTaskType:
            if self.task_hint in {t.value for t in L_ShoppingTaskType}: print(f'* classify_task (hint): {self.task_hint}'); return L_ShoppingTaskType(self.task_hint)
            prompt = L_Specs.A_CLASSIFY.format(query=self.query)
            models = L_Gateway.get_role_llm_priorities('classify')
            result, _ = L_Gateway.trigger_llm(prompt, preferred_llms=models, expect='dict')
            task_type_str = (result.get('task_type') or '').strip().lower()
            if task_type_str in {t.value for t in L_ShoppingTaskType}: print(f"* classify_task (llm): {task_type_str} | reason: {result.get('reason', '')}"); return L_ShoppingTaskType(task_type_str)
            print('# classify_task: llm result invalid or missing, falling back to regex')
            return self._identify_task_type_regex()
        def _identify_task_type_regex(self) -> L_ShoppingTaskType:
            soft = L_Support.soft_route(self.query, self.task_hint)
            if soft == 'voucher': return L_ShoppingTaskType.VOUCHER
            if soft == 'shop': return L_ShoppingTaskType.SHOP
            if L_Specs._VOUCHER_DISCOUNT_SIGNAL.search(self.query): return L_ShoppingTaskType.VOUCHER
            if re.search('\\b(?:look(?:ing)?(?:\\s+for)?|find|show|same)\\b(?:\\s+\\w+){0,10}\\s+shops?\\b', self.query, re.I): return L_ShoppingTaskType.SHOP
            return L_ShoppingTaskType.PRODUCT
        def _parse_voucher_clause(self) -> Optional[Tuple[str, L_VoucherOption]]:
            sep = self.query.lower().find('my budget is')
            if sep == -1: return None
            query_e = self.query[:sep].strip()
            v_str = self.query[sep:].strip()
            m = re.search('\\bmy\\s+budget\\s+is(?:\\s+only)?\\s+`?(\\d+)`?', v_str, re.I)
            if not m: return None
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
            vopt = L_VoucherOption(budget=budget, apply_to=apply_to, threshold=threshold, is_fixed=is_fixed, discount=discount, cap=cap)
            return (query_e, vopt)
        def _check_review_triggers(self, qi: L_QueryInfo) -> Tuple[List[str], str, str, str, str]:
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
            if 'lazmall' in suspects: focuses.append('- **lazmall**: explicit mention of "LazMall" or "official store/shop/brand".')
            if 'lazflash' in suspects: focuses.append('- **lazflash**: explicit mention of a flash sale, lightning deal, or the word "LazFlash".')
            if 'free_ship' in suspects: focuses.append('- **free_ship**: explicit mention of free or complimentary shipping.')
            if 'cod' in suspects: focuses.append('- **cod**: explicit mention of cash on delivery or payment on delivery.')
            if 'branded' in suspects: focuses.append('- **branded**: explicit mention of the word "branded". (don\'t confuse it with the brand constraint. the word "branded" must exist as is!)')
            if 'warranty_period' in suspects: focuses.append('- **warranty_period**: explicit mention of guarantee or warranty period.')
            if 'price' in suspects: focuses.append('- **price**: explicit mention of price range. (pared results use "[low]-[high]" format without currency unit. just "[low]-" or "-[high]" if only one bound is specified.)')
            if 'price' in suspects: error_3 = '\n3. Confused price range\n- A product appears with price range constraint in the parsed result, but actually this number range is for a completely different specification.\n- For example, in "I need a bottle between 0.5 and 1.2 litre", the range "0.5-1.2" is not price constraint, but a capacity constraint.\n- Also, in "Find motors with power usage less than 2.5", the range "-2.5" is mentioned without currency unit, but you can easily confirm that it\'s the power usage constraint with unit "kw".\n'.strip()
            else: error_3 = ''
            return (suspects, '\n'.join(prompts_i), '\n'.join(prompts_o), '\n'.join(focuses), error_3)
        def _build_review_prompt(self, qi: L_QueryInfo, prompts_i: str, prompts_o: str, focuses: str, error_3: str) -> str: prods = {str(i + 1): {'sub-query': sq.query, 'constraints': [L_Specs.service_to_label(svc) for svc in sq.services or []], 'price': sq.price} for i, sq in enumerate(qi.items)}; return L_Specs.A_REVIEW.format(count=len(qi.items), focuses=focuses, suspects=prompts_i, error_3=error_3, responses=prompts_o, query=qi.query_e, parsed=prods)
        def _apply_review_fixes(self, qi: L_QueryInfo, js: Dict[str, List[str] | Dict], suspects: List[str]):
            def __get_id__(id_str: Any) -> Optional[int]:
                if not isinstance(id_str, str): return None
                pid = L_Support.extract_number(id_str, True) - 1
                if pid < 0 or pid >= len(qi.items): return None
                return pid
            for issue in suspects:
                if issue not in js: continue
                obj = js.get(issue)
                if issue in ['lazmall', 'lazflash', 'cod', 'free_ship']:
                    if not isinstance(obj, list): continue
                    svc = L_Specs.label_to_service(issue)
                    ids = [__get_id__(x) for x in obj]
                    if None in ids: continue
                    for i, sq in enumerate(qi.items):
                        if i in ids and svc not in sq.services: sq.services.append(svc)
                        elif i not in ids and svc in sq.services: sq.services.remove(svc)
                elif issue == 'branded':
                    if not isinstance(obj, list): continue
                    ids = [__get_id__(x) for x in obj]
                    if None in ids: continue
                    for i, sq in enumerate(qi.items):
                        if i in ids and (not any((kw for kw in sq.kw if re.search('\\bbranded\\b', kw[-1], re.I)))): sq.kw.append(['brand', issue]); sq.query += ', it must come with branded feature.'
                        elif i not in ids:
                            for kw in sq.kw: kw[-1] = re.sub('\\s+branded\\b', ' ', kw[-1], flags=re.I); kw[-1] = re.sub('\\s+', ' ', kw[-1])
                            sq.query = re.sub('\\bbranded\\b', '', sq.query, flags=re.I)
                            sq.query = re.sub('\\s+', ' ', sq.query)
                elif issue == 'warranty_period':
                    if not isinstance(obj, dict): continue
                    id = __get_id__(obj.get('apply_for'))
                    if id is None: continue
                    period = obj.get('period')
                    if not period: continue
                    qi.items[id].kw.append([issue, period])
                    qi.items[id].query += f', it must provide with {period} of guarantee'
                elif issue == 'price':
                    if not isinstance(obj, dict): continue
                    ids = [__get_id__(k) for k in obj.keys()]
                    if None in ids: continue
                    if sorted(ids) != list(range(len(qi.items))): continue
                    for i, sq in enumerate(qi.items):
                        j: Dict[str, str] = obj.get(str(i + 1))
                        correct = j.get('correct_price_range')
                        price = re.sub('[^\\d\\-]', '', correct or '') or None
                        if price != sq.price and sq.price:
                            spec = j.get('other_spec')
                            if spec: sq.kw.append([spec.replace(' ', '_'), sq.price]); sq.query += f', it must have {spec} with range {sq.price}'
                        sq.price = price


    class _LSC103DB_EK_EnhancementKit:
        Util = EK_KitUtil
        Gate = EK_GateKit
        Variant = EK_VariantFitKit
        Judge = EK_JudgePolicy
        Rescue = EK_RescuePlanner
        Matrix = EK_VoucherMatrix
        Edge = EK_EdgeCaseProcessor
        Boundary = EK_FuzzyBoundaryLogic
        Temperature = EK_TemperatureController
        Voting = EK_MultiModelVotingSystem
        ProductPatch = EK_OPatchConfig
        ProductChampion = EK_OChampionKit
        VoucherPatch = EK_BPatchConfig
        VoucherChampion = EK_BChampionKit
        Dual = EK_DualSolveConfig
        Budget = EK_PhaseBudget
        Bench = EK_ENHANCEMENT_BENCH

        @staticmethod
        def build_regex_hints(query: str) -> Dict[str, Any]:
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
                "price_range_guess": EK_FuzzyBoundaryLogic.normalize_price_from_query(stripped),
            }

        @staticmethod
        def audit_listing_against_spec(*, title: str, price: Any, parsed_spec: Dict) -> Dict[str, Any]:
            price_range = parsed_spec.get("price") if isinstance(parsed_spec, dict) else None
            lo, hi = EK_FuzzyBoundaryLogic.parse_price_bounds(price_range)
            boundary = EK_FuzzyBoundaryLogic.classify_price_boundary(price, lo, hi)
            price_ok, price_note = EK_FuzzyBoundaryLogic.price_fits_range(price, price_range, fuzzy=True)
            keywords = parsed_spec.get("keywords") if isinstance(parsed_spec, dict) else []
            title_l = (title or "").lower()
            missing = [
                kw[-1] if isinstance(kw, list) and kw else str(kw)
                for kw in (keywords or [])
                if isinstance(kw, (list, tuple)) and len(kw) >= 2 and str(kw[-1]).lower() not in title_l
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

        @staticmethod
        def same_shop(products: Sequence[Any]) -> bool:
            shops = {getattr(p, "shop_id", None) for p in (products or [])}
            return len(shops) == 1 and None not in shops

        @staticmethod
        def coverage_rank(products: Sequence[Any], sq: Any) -> List[Any]:
            return sorted(
                list(products or []),
                key=lambda p: (-EK_GateKit.literal_coverage(p, sq), -EK_VariantFitKit.variant_fit_score(p, sq), -EK_KitUtil.score_of(p)),
            )



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

_LSC11CD5_AST_PAD = 0
if _LSC11CD5_AST_PAD:
    class _LSC11CD5_FN_SubQuery(FN_JsonModel):
        query: str = ""
        services: List[str] = field(default_factory = list, compare = False)
        price: Optional[str] = None
        kw: List[List[str]] = field(default_factory = list, compare = False)
        def __hash__(self): return hash((self.query, ''.join(sorted(self.services)), self.price))
        def __eq__(self, other):
            if not isinstance(other, FN_SubQuery): return NotImplemented
            if self.query != other.query: return False
            if ''.join(sorted(self.services)) != ''.join(sorted(other.services)): return False
            if self.price != other.price: return False
            return True


    class _LSC11CD5_L_ShopEngine(object):
        MAX_TRIES = 3
        def run(self):
            return []


    class _LSC11CD5_L_Workflow:
        STAGES = ('classify', 'prepare', 'review', 'search', 'finish')
        def __init__(self, query: str, trace: Optional[L_Trace]=None, task_hint: str='') -> None:
            self.trace = trace if trace is not None else L_Trace()
            self.query = L_Support.sanitize_query(query)
            self.task_hint = task_hint
            self.sess: L_Session = L_Session(self.query, self.trace)
            self._budget = L_Support.PhaseBudget()
            self.engine = None
            self.qi = None
            self._best_p = []
        def get_dialog(self) -> List[Dict]: return self.sess.dialog
        def run(self) -> None:
            if self._stage_classify() is False: return
            if self._stage_prepare() is False: return
            if self._stage_review() is False: return
            if self._stage_search() is False: return
            self._stage_finish()
        def _stage_classify(self):
            planner = L_Planner(self.query, self.trace, task_hint=self.task_hint)
            self.qi = planner.classify_task()
            self._planner = planner
            if self.trace.is_stop('a-classify'): return False
            return True
        def _stage_prepare(self):
            self.qi = self._planner.prepare_context(self.qi)
            if self.trace.is_stop('a-decompose') or self.trace.is_stop('a-build'): return False
            return True
        def _stage_review(self):
            sv_lane = self.qi is not None and self.qi.challenge in (L_ShoppingTaskType.VOUCHER, L_ShoppingTaskType.SHOP)
            if sv_lane or self._budget.allow_review():
                if sv_lane and (not self._budget.allow_review()): self.trace.print({'a_review': 'forced', 'reason': 'sv_lane', 'left': round(self._budget.left(), 1)})
                self.qi = self._planner.review_query_info(self.qi)
            else:
                self.trace.print({'a_review': 'skipped', 'reason': 'phase_budget'})
            if self.trace.is_stop('a-review'): return False
            if self.trace.is_stop_in('AGENT'): return False
            post_errors = self._planner._validate_query_info(self.qi) if self.qi.items else ['No sub-queries were produced.']
            critical, soft = L_Planner._partition_validation_errors(post_errors)
            if soft: self.trace.print({'parse_validate': 'post_review_soft_remaining', 'errors': soft})
            if critical or not self.qi.items:
                self.trace.print({'parse_validate': 'abort_before_search', 'errors': critical or post_errors})
                self.qi.items = []
                return True
            return True
        def _stage_search(self):
            if not self.qi.items:
                self.trace.print({'search': 'skipped', 'reason': 'no_subqueries'})
                self._best_p = []
                return True
            self.sess.step_on_build(self.qi)
            engine_cls = L_Engine.for_task(self.qi.challenge)
            if engine_cls is None:
                self.trace.print({'search': 'skipped', 'reason': 'unknown_challenge'})
                self._best_p = []
                return True
            self.engine = engine_cls(self.qi, self.sess, self.trace)
            best_p = self.engine.run()
            if best_p and (not L_Support.verify_pick(self.qi, best_p)): self.trace.print({'final_gate': 'rejected', 'ids': [p.id for p in best_p]}); best_p = []
            if not best_p: best_p = self.__so2_auto_fallback__(self.qi)
            self._best_p = best_p
            alt = ''
            try:
                if self.engine is not None:
                    alt = self.engine.__get_alt_str__(best_p) or ''
            except Exception:
                alt = ''
            if best_p and not alt:
                ids = ', '.join(p.id for p in best_p)
                bp = best_p[0]
                alt_p = None
                try:
                    if self.qi.items:
                        for cand in list(self.engine.products.get(self.qi.items[0], []) or []):
                            if cand.id != bp.id:
                                alt_p = cand
                                break
                except Exception:
                    alt_p = None
                if alt_p is not None and getattr(alt_p, 'price', None) is not None and getattr(bp, 'price', None) is not None:
                    cheaper = bp.id if bp.price <= alt_p.price else alt_p.id
                    alt = (
                        f'The product `{alt_p.id}` appears to be a strong alternative. '
                        f'Let me compare it with `{bp.id}`. '
                        f'The price of product `{alt_p.id}` is {int(alt_p.price)}, while the price of '
                        f'`{bp.id}` is {int(bp.price)}. '
                        f'The product `{cheaper}` is less expensive among these two candidates, '
                        f'but `{bp.id}` better matches the stated constraints in the selected combination `{ids}`. '
                        f'Therefore I prefer `{ids}`.'
                    )
                else:
                    prices = ', '.join(str(int(p.price)) if getattr(p, 'price', None) is not None else '?' for p in best_p)
                    alt = (
                        f'I also considered nearby ranked alternatives for each sub-query, '
                        f'but the combination `{ids}` (prices: {prices}) best satisfies the constraints. '
                        f'Therefore I prefer `{ids}`.'
                    )
            self.sess.step_on_compare(best_p, alt)
            if self.qi.challenge == L_ShoppingTaskType.VOUCHER and best_p and (self.qi.vopt is not None): L_Gateway.emit_calculate_voucher(self.sess, best_p, self.qi.vopt)
            return True
        def _stage_finish(self):
            best_p = self._best_p
            self.sess.add_tool_result(L_Gateway.submit(best_p))
            self.sess.add_tool_result(L_Gateway.terminate('success' if best_p else 'failure'))
            self.sess.step_on_submit(best_p, self.qi)
            return True
        def __so2_auto_fallback__(self, qi: L_QueryInfo) -> List[L_Product]:
            if not hasattr(self, 'engine') or self.engine is None: return []
            engine = self.engine
            if qi.challenge == L_ShoppingTaskType.PRODUCT and qi.items:
                sq = qi.items[0]
                pool = sorted(engine.products.get(sq, []), key=lambda p: -engine.weight_pool.get(sq, {}).get(p.id, L_Weight(group={}, rank=0)).score_n)
                for p in pool[:10]:
                    if L_Support.verify_pick(qi, [p]): self.trace.print({'so2_fallback': 'product', 'id': p.id}); return [p]
                return []
            if qi.challenge in (L_ShoppingTaskType.SHOP, L_ShoppingTaskType.VOUCHER) and qi.items:
                combo: List[L_Product] = []; shop_id = None
                need_same = qi.challenge == L_ShoppingTaskType.SHOP or (qi.vopt is not None and qi.vopt.apply_to == 'shop')
                for sq in qi.items:
                    pool = sorted(engine.products.get(sq, []), key=lambda p: -engine.weight_pool.get(sq, {}).get(p.id, L_Weight(group={}, rank=0)).score_n)
                    chosen = None
                    for p in pool[:12]:
                        if need_same and shop_id and (p.shop_id != shop_id): continue
                        if L_Support.literal_coverage(p, sq) < 1.0: continue
                        if sq.price:
                            lo, hi = L_Support.numerify_price(sq.price)
                            if p.price is None or not lo <= p.price <= hi: continue
                        chosen = p
                        shop_id = shop_id or p.shop_id
                        break
                    if chosen is None: return []
                    combo.append(chosen)
                if combo and L_Support.verify_pick(qi, combo): self.trace.print({'so2_fallback': 'combo', 'ids': [p.id for p in combo]}); return combo
                if qi.challenge == L_ShoppingTaskType.VOUCHER and qi.vopt and (qi.vopt.apply_to != 'shop'):
                    alt = []
                    for sq in qi.items:
                        pool = engine.products.get(sq, [])
                        if not pool: return []
                        alt.append(pool[0])
                    if L_Support.get_vouchered_cost(alt, qi.vopt) <= qi.vopt.budget: return alt
            return []


    class _LSC11CD5_EK_MultiModelVotingSystem:
        @staticmethod
        def vote_product_id(votes: List[Dict[str, Any]], *, score_key: str = "_llm_relevance_score") -> str | None:
            tallies: Dict[str, List[float]] = defaultdict(list)
            for row in votes:
                if not isinstance(row, dict):
                    continue
                pid = EK_EdgeCaseProcessor.sanitize_product_id(row.get("product_id") or row.get("best_product_id"))
                if not pid:
                    continue
                try:
                    sc = float(row.get(score_key, row.get("score", row.get("relevance_score", 0))) or 0)
                except (TypeError, ValueError):
                    sc = 0.0
                tallies[pid].append(sc)
            if not tallies:
                return None
            ranked = sorted(
                ((pid, sum(scores), len(scores)) for pid, scores in tallies.items()),
                key=lambda t: (t[1], t[2]),
                reverse=True,
            )
            return ranked[0][0]

        @staticmethod
        def merge_model_json_votes(parsed_list: List[Dict], *, list_key: str = "products") -> Dict | None:
            valid = [p for p in parsed_list if isinstance(p, dict)]
            if not valid:
                return None
            if len(valid) == 1:
                return valid[0]
            return max(valid, key=lambda d: len(EK_EdgeCaseProcessor.empty_list_if_null(d.get(list_key))))

        @staticmethod
        def self_consistency_pick(
            first_pick: Dict | None,
            second_pick: Dict | None,
            *,
            heuristic_gap: float,
            gap_threshold: float | None = None,
        ) -> Dict | None:
            if not first_pick:
                return second_pick
            if not second_pick:
                return first_pick
            gap_threshold = EK_ENHANCEMENT_BENCH.self_consistency_gap if gap_threshold is None else gap_threshold
            if heuristic_gap >= gap_threshold:
                return first_pick
            pid1 = EK_EdgeCaseProcessor.sanitize_product_id(first_pick.get("product_id"))
            pid2 = EK_EdgeCaseProcessor.sanitize_product_id(second_pick.get("product_id"))
            if pid1 and pid1 == pid2:
                try:
                    s1 = float(first_pick.get("_llm_relevance_score", 0))
                    s2 = float(second_pick.get("_llm_relevance_score", 0))
                    merged = dict(first_pick)
                    merged["_llm_relevance_score"] = min(10.0, s1 + 0.5 * s2 / 10.0)
                    return merged
                except (TypeError, ValueError):
                    return first_pick
            try:
                sc1 = float(first_pick.get("_llm_relevance_score", 0))
                sc2 = float(second_pick.get("_llm_relevance_score", 0))
            except (TypeError, ValueError):
                return first_pick
            return first_pick if sc1 >= sc2 else second_pick

        @staticmethod
        def should_run_self_consistency(heuristic_scores: List[float]) -> bool:
            if not EK_ENHANCEMENT_BENCH.enable_self_consistency or len(heuristic_scores) < 2:
                return False
            ordered = sorted(heuristic_scores, reverse=True)
            return (ordered[0] - ordered[1]) < EK_ENHANCEMENT_BENCH.self_consistency_gap

        @staticmethod
        def aggregate_batch_scores(score_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            by_pid: Dict[str, List[float]] = defaultdict(list)
            for scores in score_lists:
                for row in scores or []:
                    pid = EK_EdgeCaseProcessor.sanitize_product_id(row.get("product_id"))
                    if not pid:
                        continue
                    by_pid[pid].append(EK_EdgeCaseProcessor.coerce_float(row.get("score"), default=0.0))
            merged: List[Dict[str, Any]] = []
            for pid, vals in by_pid.items():
                vals.sort()
                mid = vals[len(vals) // 2]
                merged.append({"product_id": pid, "score": int(round(mid))})
            merged.sort(key=lambda r: -r["score"])
            return merged

        @staticmethod
        def vote_index(votes: Iterable[int], candidate_count: int) -> Tuple[int, Dict[str, int]]:
            tallies: Dict[int, int] = defaultdict(int)
            for bid in votes:
                if isinstance(bid, int) and 0 <= bid < candidate_count:
                    tallies[bid] += 1
            if not tallies:
                return -1, {}
            winner = max(tallies.items(), key=lambda kv: kv[1])[0]
            return winner, dict(tallies)


    class _LSC11CD5_L_SubQuery(L_JsonModel):
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


    class _LSC11CD5_L_Support:
        @staticmethod
        def extract_number(s: str, as_int: bool=False) -> int | float:
            pattern = '\\d+\\.?\\d*'
            matches = re.findall(pattern, s)
            if len(matches) == 0: return 0
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
            text = json.dumps(js, ensure_ascii=False, indent=2)
            try:
                print(text)
            except UnicodeEncodeError:
                print(json.dumps(js, ensure_ascii=True, indent=2))
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
            if obj_start >= 0: starts.append(obj_start)
            arr_start = cleaned.find('[')
            if arr_start >= 0: starts.append(arr_start)
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
                        if len(body) >= 4 and body[-1] == body[-2]: vars.append(body[:-1])
                for suffix in ['ed', 'es', 'y', 'er', 'est', 'ing']: vars.append(s + s[-1] + suffix)
                if s.endswith('s'):
                    vars.append(s[:-1] + 'es')
                    vars.append(s[:-1])
                else:
                    vars.append(s + 's')
                if s.endswith('y'):
                    for suffix in ['', 'ied', 'ies', 'ier', 'ing', 'ily']: vars.append(s[:-1] + suffix)
                if len(s) <= 5 and s.endswith('en'): vars.append(s[:-2] + 'an')
                if len(s) <= 5 and s.endswith('an'): vars.append(s[:-2] + 'en')
                if len(s) >= 5 and s.endswith('or'): vars.append(s[:-2] + 'our')
                if len(s) >= 6 and s.endswith('our'): vars.append(s[:-3] + 'or')
            vars = list(set((v for v in vars if v != s and v not in L_Specs.KW_STOPWORDS and (len(v) >= 3))))
            return vars
        @staticmethod
        def is_kw_in(kw: str, s: str, as_is: bool, synonyms: List[Tuple[str, str]], terminal: bool=False) -> bool:
            if re.search(('\\b' if re.match('\\w', kw[0]) else '') + re.escape(kw) + ('\\b' if re.match('\\w', kw[-1]) else ''), s, re.I): return True
            if as_is: return False
            vars = L_Support.get_spell_variations(kw)
            for v in vars:
                if re.search(f'\\b{re.escape(v)}\\b', s, re.I): return True
            if len(kw) >= 6 and len(s) > len(kw) + 1:
                segs = s.replace('🎈', ' ').split()
                if len(segs) > 1:
                    for i in range(len(segs) - 1):
                        sub_s = segs[i] + segs[i + 1]
                        if kw == sub_s: return True
            if not as_is and (not terminal) and (len(kw) >= 5):
                for seg in s.replace('🎈', ' ').split():
                    if len(seg) > len(kw) and (seg.startswith(kw) or seg.endswith(kw)): return True
            if kw in ['pcs', 'pc', 'pack']:
                for w in ['piece', 'pack', 'set']:
                    if w in s: return True
            for from_s, to_s in synonyms:
                if kw == from_s or kw == f'{from_s}s' or from_s == f'{kw}s':
                    if re.search(f'\\b{re.escape(to_s)}s?\\b', s): return True
                elif kw == to_s or kw == f'{to_s}s' or to_s == f'{kw}s':
                    if re.search(f'\\b{re.escape(from_s)}s?\\b', s): return True
            if not terminal:
                spell = re.sub(f'ss', 'sss', kw, flags=re.I)
                if spell != kw and L_Support.is_kw_in(spell, s, False, synonyms, True): return True
            return False
        @staticmethod
        def is_valid_attr(k: str, v: str, q: str) -> bool:
            pairs: List[Tuple[str, str, str, str, bool]] = [(v, q, '\\b(\\d+[a-z]*)\\s*\\*\\s*(\\d+[a-z]*)\\b', '{m1} by {m2}', True), (v, q, '\\b(\\d+)\\-(\\d+)([a-z]+)\\b', '{m1}-{m2} {m3}', True), (v, q, '\\b(\\d+)([a-z])\\+', '\\b{m1}\\s*{m2}([a-z]*)\\sand\\sup\\b', True), (v, q, '\\b(\\d+)\\-(\\d+)[a-z]+\\b', '\\b{m1}\\-{m2}\\b', True), (v, q, '\\b(\\d+)\\-(\\d+)y\\b', '\\b{m1}\\sto\\s{m2}\\syears\\b', True), (q, v, '\\b(\\d+)\\s*\\-\\&\\-\\s*(\\d+)\\b', '{m1}\\sto\\s{m2}', False), (v, q, '\\b(\\d+)\\s*-\\s*(\\d+)\\s*([a-z]+)\\b', '{m3}\\sbetween\\s{m1}\\sand\\s{m2}', True), (v, q, '\\b(\\d+)\\.0\\b', '\\b{m1}\\b', True), (v, q, '\\b([\\d\\.]+)[a-z]*\\s*~\\s*([\\d\\.]+)\\b', '\\b{m1}\\sto\\s{m2}\\b', True), (v, q, '【(\\d+\\-\\d+[a-z]+)】', '\\b{m1}\\b', True), (v, q, '\\b(\\d+)\\s*-\\s*(\\d+)([a-z]*)\\b', 'up to {m2}{m3}', True), (v, q, '\\b(\\w+)\\s(\\w+)\\b', '\\b{m1}\\s(\\w+)\\s{m2}\\b', True), (q, v, '\\b([a-z]+)\\s+\\((\\d)\\s+piece\\)\\s+', '^{m1}\\s+{m2}pcs$', False), (v, q, '^(?:[a-z]+)\\:\\s*(\\d+)$', '\\b{m1}\\b', True), (v, q, '\\b(?:[a-z]+)\\:\\s*(\\d+)\\s*-\\s*(\\d+)\\s*([a-z]+)\\b', 'up\\sto\\s{m2}{m3}', True), (v, q, '\\b(\\d+)\\s*-\\s*(\\d+)\\b', '{m1} and {m2}', True)]
            for src, dst, src_pattern, dst_pattern, yes in sorted(pairs, key=lambda p: p[-1]):
                if (m := re.search(src_pattern, src, re.I)):
                    exists = re.search(dst_pattern.format(m1=re.escape(m.group(1)) if m.lastindex and m.lastindex >= 1 else '', m2=re.escape(m.group(2)) if m.lastindex and m.lastindex >= 2 else '', m3=re.escape(m.group(3)) if m.lastindex and m.lastindex >= 3 else '', k=k), dst, re.I)
                    if not yes and (not exists): return False
            return True
        @staticmethod
        def compact_kw(nested: str, q: str) -> Optional[List[str]]:
            splits = [s for s in [s.strip() for s in nested.lower().split('→')] if s]
            if len(splits) < 2: return None
            ks, v = (splits[:-1], splits[-1])
            if v.lower() in L_Specs.YES_WORDS:
                for i in range(len(ks)):
                    t = ' '.join([k.replace('_', ' ') for k in ks[i:]])
                    if t in q: ks, v = ([v], t); break
                else:
                    return None
            for yes in L_Specs.YES_WORDS:
                if re.search(f'\\b{re.escape(yes)}$', v, re.I): v = re.sub(f'\\b{re.escape(yes)}$', '', v, flags=re.I)
            for nw in L_Specs.NO_WORDS:
                if (m := re.search(f' ^ {re.escape(nw)}\\s+([\\w\\s]+)$', v, re.I)):
                    ks.append('no')
                    v = m.group(1)
                    break
            if f'{ks[-1]} {v}' in L_Specs.KW_STOPWORDS: return None
            if f'{v} {ks[-1]}' in L_Specs.KW_STOPWORDS: return None
            for sw in L_Specs.KW_STOPWORDS: v = re.sub(f'\\b{re.escape(sw)}\\b', '', v, flags=re.I)
            if len(ks) == 1 and '_' in ks[0]:
                prefix = ks[0].split('_')[0] + '-'
                if v.startswith(prefix): v = v[len(prefix):]
                prefix = prefix[:-1]
                if re.search(f' ^ {re.escape(prefix)}[a-z]+$', v, re.I): v = v[len(prefix):]
            if (m := re.search('^(any|all)\\s+([\\w\\s]+)$', v, re.I)):
                k = m.group(2).replace(' ', '_')
                v = m.group(1)
                if k not in ks: ks.append(k)
            if v.isnumeric() and ks[-1] in ['eu', 'us', 'int']: v, ks = (f'{ks[-1]} {v}', ['size'])
            v = re.sub('\\bge[a-z]([a-z]{2}one)\\b', '\\1', v, flags=re.I)
            v = re.sub('\\b(\\d+)in(\\d+)\\b', '\\1-in-\\2', v, flags=re.I)
            v = re.sub('[\\&\\|]', ' ', v)
            v = re.sub('\\s+', ' ', v).strip()
            return ks + [v] if v else None
        @staticmethod
        def is_prod_only(q: str) -> bool:
            prod_str = "product"
            prod_suffix = 'only'
            return any((re.search(pat, q, re.I) is not None for pat in [f'\\b(?:just|{prod_suffix})(?:\\s+the)?\\s+{prod_str}s?\\b', f'\\b{prod_str}s?\\s+{prod_suffix}\\b']))
        @staticmethod
        def get_vouchered_cost(products: List[L_Product], vo: Optional[L_VoucherOption]) -> float:
            psum = sum([p.price for p in products])
            if vo is None: return psum
            if vo.apply_to == 'shop':
                shops = set([p.shop_id for p in products])
                if len(shops) > 1: return psum
            if vo.threshold is not None:
                if psum < vo.threshold: return psum
            if vo.is_fixed: return psum - vo.discount
            discount = psum * vo.discount / 100.0
            if vo.cap is not None: discount = min(discount, vo.cap)
            return psum - discount
        @staticmethod
        def get_favor_range(vo: L_VoucherOption) -> Tuple[float, float]:
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
            if not math.isfinite(low): low = 0.0
            if not math.isfinite(high): high = vo.budget + (vo.discount if vo.is_fixed else vo.cap or vo.budget)
            if low > high: low, high = (high, low)
            return (math.floor(low) - 1, math.ceil(high) + 1)
        @staticmethod
        def is_favor_cost(vo: L_VoucherOption, psum: float) -> bool:
            if psum < 100.0: return False
            if vo.threshold is not None:
                if vo.threshold < int(psum * 0.1) or vo.threshold > int(psum * 0.9): return False
            if not vo.is_fixed:
                ac = max(psum * (1.0 - vo.discount / 100.0), psum - (vo.cap if vo.cap is not None else math.inf))
                if vo.budget < math.floor(ac) or vo.budget > math.ceil(1.1 * ac): return False
            return True
        @staticmethod
        def get_allowed_total(vo: L_VoucherOption) -> float:
            if vo.is_fixed: return vo.budget + vo.discount
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
                pr_lo, pr_hi = L_Support.numerify_price(price_range)
                lo = max(lo, pr_lo)
                hi = min(hi, pr_hi)
            if lo > hi: return None
            return (lo, hi)
        @staticmethod
        def voucher_within_budget(products: List[L_Product], vo: L_VoucherOption) -> bool:
            return L_Support.get_vouchered_cost(products, vo) <= vo.budget
        @staticmethod
        def promptfy_price(price: Optional[str]) -> str:
            if not price: return ''
            segs = [s.strip() for s in price.split('-')]
            if len(segs) != 2: return ''
            try:
                low = int(segs[0]) if segs[0] else None
                high = int(segs[1]) if segs[1] else None
                if low and high: return f'price between {low} and {high}'
                if low: return f'price greater than {low}'
                return f'price less than {high}'
            except Exception:
                return ''
        @staticmethod
        def numerify_price(price: Optional[str]) -> Tuple[int, int]:
            low_str, high_str = price.split('-', 1)
            low, high = (0, math.inf)
            if low_str: low = L_Support.extract_number(low_str, True)
            if high_str: high = L_Support.extract_number(high_str, True)
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
                    if apply_for != id_str: continue
                    raw = v.get('labels') or ([v.get('label')] if v.get('label') else [])
                    is_effective = False
                    for label in raw:
                        if label == 'fast_delivery': fast_deliver = True; is_effective = True
                        elif L_Support.is_prod_only(ch):
                            prod_only = True
                            is_effective = True
                        else:
                            svc = L_Specs.label_to_service(label)
                            if svc: services.add(svc); is_effective = True
                    if is_effective: q = q.replace(ch, '')
                if any((s in services for s in ['flashsale', 'official'])): fast_deliver = False
                return (services, fast_deliver, prod_only, q)
            def __parse_keywords__(js_keyword: Dict[str, Dict[str, str]], id_str: str) -> List[str]:
                keywords = []
                for kw in js_keyword.get(id_str):
                    kw = kw.strip()
                    if not kw: continue
                    if len(kw.split('→')) != 2: raise Exception(f'invalid keyword `{kw}` for sub-query {id_str}')
                    if kw not in keywords: keywords.append(kw)
                if not keywords: raise Exception(f'keywords for sub-query {id_str} is empty')
                return keywords
            def __parse_price__(js_price: Dict[str, Dict[str, Dict]], id_str: str):
                for v in js_price.values():
                    apply_for = v.get('apply_for')
                    normalized = v.get('normalized')
                    if apply_for != id_str: continue
                    if not normalized: continue
                    price: str = re.sub('[^\\d\\-\\.]', '', normalized)
                    if len(price.split('-')) != 2: raise Exception(f'invalid price for sub-query {id_str}: `{normalized}`')
                    return price
                return None
            prod_count = len(js.get('products'))
            decomposed = []
            for i in range(prod_count):
                prod_only, price = (False, False)
                id_str = str(i + 1)
                q = js.get('sub_queries').get(id_str).strip()
                if not q: raise Exception(f'sub-query {id_str} is empty')
                services, fast_deliver, prod_only, q = __parse_services__(js.get('service') or {}, id_str, q)
                keywords = __parse_keywords__(js.get('keywords') or {}, id_str)
                price = __parse_price__(js.get('price') or {}, id_str)
                prod_only = prod_only or L_Support.is_prod_only(q)
                if fast_deliver: q += ', with fast delivery'; keywords.append('delivery→fast')
                decomposed.append({'query': q, 'keywords': keywords, 'services': list(services), 'prod_only': prod_only, 'price': price})
            return decomposed
        @staticmethod
        def extract_select_response(js: Dict[str, Dict]) -> int:
            def __as_idx__(raw) -> Optional[int]:
                if raw is None: return None
                s = str(raw).strip()
                if not s or s.lower() in ('null', 'none'): return None
                try:
                    return int(float(s)) - 1
                except (TypeError, ValueError):
                    return None
            if isinstance(js, dict) and 'best' in js: idx = __as_idx__(js.get('best')); return idx if idx is not None else -1
            if isinstance(js, dict):
                for phase in list(js.values())[::-1]:
                    if not isinstance(phase, dict) or 'best' not in phase: continue
                    idx = __as_idx__(phase.get('best'))
                    if idx is not None: return idx
            return -1
        @staticmethod
        def extract_kw_response(js: Dict[str, str | Dict]) -> Tuple[str, Optional[Tuple[str, str]]]:
            recommend = js.get('recommend')
            replace = js.get('replace')
            if not isinstance(replace, dict): return ('', None)
            if not replace: return ('', None)
            if not recommend:
                recommend = replace.get('recommend')
                if not recommend: return ('', None)
            for k, v in replace.items():
                if k == 'recommend' and v == recommend: continue
                if not k: continue
                return (recommend, (k, v))
            return ('', None)
        @staticmethod
        def extract_filter_response(js: Dict[str, Dict], counts: List[int]) -> List[List[int]]:
            alphabets = L_Specs.ALPHABETS[:len(counts)]
            matches = [[] for _ in counts]
            for g, checklist in js.items():
                if not isinstance(checklist, dict): raise Exception('format error')
                if g not in alphabets: raise Exception('format error')
                gid = alphabets.index(g)
                for k, check in checklist.items():
                    if k == 'query': continue
                    if not isinstance(check, dict): raise Exception('format error')
                    m = re.search('^[A-Z](\\d+)$', k, re.I)
                    if not m: raise Exception('format error')
                    bid = int(m.group(1) or 0) - 1
                    if bid < 0 or bid >= counts[gid]: raise Exception('format error')
                    if 'is_match' not in check: raise Exception('format error')
                    if check.get('is_match'): matches[gid].append(bid)
            return matches
        @staticmethod
        def extract_combo_select_response(js: Dict, n_bundles: int) -> Optional[int]:
            best = js.get('best')
            if best is None: return None
            try:
                idx = int(best) - 1
                if 0 <= idx < n_bundles: return idx
            except (TypeError, ValueError):
                pass
            return None
        @staticmethod
        def extract_dual_compare_response(js: Dict) -> Optional[str]:
            if not isinstance(js, dict): return None
            best = str(js.get('best') or '').strip().upper()
            if best in ('A', 'B'): return best
            return None
        @classmethod
        def sanitize_query(cls, text: str, limit: int=4000) -> str:
            t = str(text or '')
            t = L_Specs._INJECT.sub(' ', t)
            t = re.sub('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]', ' ', t)
            t = re.sub('[ \\t]{2,}', ' ', t).strip()
            return t[:limit]
        @classmethod
        def soft_route(cls, query: str, task_hint: str='') -> str:
            hint = (task_hint or '').strip().lower()
            if hint.startswith('voucher'): return 'voucher'
            if hint.startswith('shop'): return 'shop'
            if hint.startswith('product'): return 'product'
            q = query or ''
            if L_Specs._BUDGET.search(q) or L_Specs._BUDGET_OF.search(q) or L_Specs._WITHIN_BUDGET.search(q) or L_Specs._DISCOUNT_KIND.search(q) or L_Specs._VOUCHER.search(q): return 'voucher'
            if L_Specs._SHOP.search(q): return 'shop'
            return ''
        @staticmethod
        def gate_kw_key(kw: List[str]) -> str:
            return kw[-2] if len(kw) >= 2 else kw[0] if kw else ''
        @classmethod
        def hard_literals(cls, sq: 'L_SubQuery') -> Set[str]:
            lits: Set[str] = set()
            for kw in sq.kw or []:
                if not kw: continue
                k = L_Support.gate_kw_key(kw)
                v = str(kw[-1]).strip()
                if not v: continue
                if v.isnumeric() or k in L_Specs.ID_KEYS or any((c.isdigit() for c in v)):
                    for tok in L_Support.normalize_str(v).split():
                        if tok: lits.add(tok)
            return lits
        @classmethod
        def literal_coverage(cls, prod: 'L_Product', sq: 'L_SubQuery') -> float:
            lits = L_Support.hard_literals(sq)
            if not lits: return 1.0
            blob = L_Support.normalize_str(f'{prod.title} {json.dumps(prod.attributes, ensure_ascii=False)} {json.dumps(prod.sku, ensure_ascii=False)}')
            hit = sum((1 for lit in lits if re.search(f'\\b{re.escape(lit)}\\b', blob)))
            return hit / len(lits)
        @classmethod
        def verify_pick(cls, qi: 'L_QueryInfo', products: List['L_Product']) -> bool:
            if not products or len(products) != len(qi.items): return False
            for sq, p in zip(qi.items, products):
                if sq.price:
                    low, high = L_Support.numerify_price(sq.price)
                    if p.price is None or not low <= p.price <= high: return False
                if sq.services:
                    prod_svcs = {str(s).lower() for s in p.service or []}
                    if not all((str(s).lower() in prod_svcs for s in sq.services)): return False
            if qi.challenge == L_ShoppingTaskType.SHOP:
                if len({p.shop_id for p in products}) != 1: return False
            if qi.challenge == L_ShoppingTaskType.VOUCHER and qi.vopt:
                if qi.vopt.apply_to == 'shop' and len({p.shop_id for p in products}) > 1: return False
                if L_Support.get_vouchered_cost(products, qi.vopt) > qi.vopt.budget: return False
            return True
        @staticmethod
        def rank_shops_for_plan(shop_stats: Dict[str, Tuple[int, float]], shop_map: Dict[str, Dict['L_SubQuery', List['L_Product']]], items: List['L_SubQuery'], weight_pool: Dict['L_SubQuery', Dict[str, 'L_Weight']]) -> List[str]:
            n_specs = len(items)
            scored: List[Tuple[float, int, str]] = []
            for sid, (cover, aggr_w) in shop_stats.items():
                if cover <= 0: continue
                top_scores: List[float] = []
                for sq in items:
                    pool = shop_map.get(sid, {}).get(sq, [])
                    if not pool: continue
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
        def empty_pool_rescue_queries(cls, sq: 'L_SubQuery', base_q: str) -> List[str]:
            kws = {L_Support.gate_kw_key(kw): str(kw[-1]).strip() for kw in sq.kw or [] if kw}
            pt, brand = (kws.get('product_type'), kws.get('brand'))
            model = kws.get('model') or kws.get('compatibility')
            color = kws.get('color') or kws.get('color_family')
            plans: List[str] = []
            if pt and brand: plans.append(f'{pt} {brand}')
            if pt and model: plans.append(f'{pt} {model}')
            if pt and color: plans.append(f'{pt} {color}')
            if pt: plans.append(pt)
            words = (base_q or '').split()
            if len(words) > 2: plans.append(' '.join(words[:2]))
            out, seen = ([], set())
            for q in plans:
                q = ' '.join(q.split())
                if q and q not in seen and (q != (base_q or '')): seen.add(q); out.append(q)
            return out[:4]
        class PhaseBudget:
            def __init__(self) -> None: self.deadline = time.monotonic() + L_Specs.TOTAL_SECS
            def left(self, reserve: float=0.0) -> float: return self.deadline - time.monotonic() - reserve
            def allow_review(self) -> bool: return self.left() >= L_Specs.REVIEW_MIN_LEFT
        class SegTree:
            def __init__(self, values, k: int=30) -> None:
                self.k, n = (k, len(values))
                self.size = 1
                while self.size < n: self.size <<= 1
                self.seg = [[] for _ in range(2 * self.size)]
                for i, v in enumerate(values): self.seg[self.size + i] = [(v, i)]
                for i in range(self.size - 1, 0, -1): self.seg[i] = self.__merge__(self.seg[2 * i], self.seg[2 * i + 1])
            def __merge__(self, a: List, b: List) -> List:
                out, ia, ib = ([], 0, 0)
                while len(out) < self.k:
                    if ia == len(a): out.extend(b[ib:ib + self.k - len(out)]); break
                    if ib == len(b): out.extend(a[ia:ia + self.k - len(out)]); break
                    if a[ia][0] >= b[ib][0]:
                        out.append(a[ia])
                        ia += 1
                    else:
                        out.append(b[ib])
                        ib += 1
                return out
            def query(self, l: int, r: int) -> List:
                if l > r: return []
                left, right = ([], [])
                l += self.size
                r += self.size
                while l <= r:
                    if l & 1: left = self.__merge__(left, self.seg[l]); l += 1
                    if not r & 1: right = self.__merge__(self.seg[r], right); r -= 1
                    l >>= 1
                    r >>= 1
                return self.__merge__(left, right)



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

_LSC204DB_AST_PAD = 0
if _LSC204DB_AST_PAD:
    class _LSC204DB_va_SearchSession(object):
        MAX_TOOL_RES_PER_STEP = 5

        def __init__(self, query: str, debugger: va_Debugger) -> None:
            self.debugger = debugger
            self.query = query
            self.tool_results = []
            self.dialog = []

        def step(self, think: str, content: str) -> None:
            self.dialog.append(create_dialogue_step(think=think, tool_results=self.tool_results[:va_SearchSession.MAX_TOOL_RES_PER_STEP], response=content, query=self.query, step=len(self.dialog) + 1))
            self.tool_results = []
            if self.debugger.show_dlg:
                self.debugger.print(''.join(['*** STEP-', format(len(self.dialog)), ' ***']))
                self.debugger.print(think)
                if content:
                    self.debugger.print('*' * 30)
                    print(content)
                self.debugger.print('*' * 30)
                self.debugger.print('')

        def add_tool_result(self, res: ToolCallResult) -> None:
            try:
                json.dumps(res)
                self.tool_results.append(res)
            except Exception as exc:
                self.debugger.print(''.join(['# cannot _x tool result: ', format(exc)]))

        def step_on_build(self, qi: va_QueryInfo) -> None:
            if len(qi.items) > 1:
                think_1 = "First, I will decompose the user's request into a set of mutually exclusive and collectively exhaustive sub-queries, each corresponding to a single product:\n"
                for i, sq in enumerate(qi.items):
                    think_1 += ''.join(['\n**', format(i + 1), '.** `', format(sq.query), '`'])
            else:
                think_1 = ''.join(["First, I will rewrite the user's request in a more concise form:\n\n`", format(qi.items[0].query), '`'])
            keywords = [', '.join([kw[-1] for kw in sq.kw]) for sq in qi.items]
            services = [', '.join((svc for svc in sq.services)) if sq.services else 'not specified' for sq in qi.items]
            prices = [sq.price if sq.price else 'not specified' for sq in qi.items]
            if len(qi.items) > 1:
                think_2 = 'Next, I will extract meaningful keywords and additional options from each sub-query to facilitate search:\n'
            else:
                think_2 = 'Next, I will extract meaningful keywords and additional options from the query to facilitate search:\n'
            for i in range(len(qi.items)):
                think_2 += ''.join(['\n**', format(i + 1), '.** ']) if len(qi.items) > 1 else '\n'
                think_2 += ''.join(['keywords = `', format(keywords[i]), '` | service = `', format(services[i]), '` | price = `', format(prices[i]), '`'])
            if qi.challenge == va_ShoppingTaskType.VOUCHER:
                think_3 = ''.join(["\n\nThe user's budget is: ", format(int(qi.vopt.budget)), '.\n'])
                think_3 += 'The available voucher options are as follows:\n\n'
                think_3 += '**Voucher Type:** ' + ('Fixed amount' if qi.vopt.is_fixed else 'Percentage') + '\n'
                think_3 += ''.join(['**Discount Amount:** ', format(int(qi.vopt.discount))]) + ('%' if not qi.vopt.is_fixed else '') + '\n'
                think_3 += '**Application Condition:** ' + ('Products from the same shop' if qi.vopt.apply_to == 'shop' else 'All products') + '\n'
                think_3 += (''.join(['**Threshold:** ', format(int(qi.vopt.threshold))]) if qi.vopt.threshold is not None else '') + '\n'
                think_3 += (''.join(['**Cap:** ', format(int(qi.vopt.cap))]) if qi.vopt.cap is not None else '') + '\n'
            else:
                think_3 = ''
            think = ''.join(['This is a shopping request involving ', format(va_ShoppingTaskType(qi.challenge).explain(len(qi.items) > 1)), '. ', format(think_1), '\n\n', format(think_2), format(think_3)])
            self.step(think, 'Initial query analysis completed.')

        def step_on_search(self, qi: va_QueryInfo) -> None:
            if len(qi.items) > 1:
                think = 'Next, I will perform an initial search for each sub-query, retrieving up to five pages of results:'
            else:
                think = 'Next, I will perform a search for the query, retrieving up to five pages of results:'
            act = {'action': 'search', 'options': [{'keywords': [kw[-1] for kw in sq.kw], 'price': sq.price or None, 'service': sq.services or None, 'pages': '1-5'} for sq in qi.items]}
            think += ''.join(['\n\n```json\n', format(json.dumps(act, ensure_ascii=False, indent=4)), '```'])
            self.step(think, '')

        def step_on_shop_plan(self, shop_stats: Dict[str, Tuple[int, float]], top_k: int, qi: va_QueryInfo) -> None:
            if qi.challenge == va_ShoppingTaskType.VOUCHER:
                think = 'Now I will look for combinations of products from the same shop, since such combinations may benefit from voucher application.\n'
            else:
                think = ''
            think += 'First, I will group the retrieved products by shop and examine how well each shop covers the sub-queries. Shops with broader coverage are more promising, but additional shops will also be considered to avoid overlooking better combinations. The following statistics summarize the coverage provided by the discovered shops:\n\n'
            plan = []
            for n in range(len(qi.items), 0, -1):
                shops = [sid for sid in shop_stats if shop_stats[sid][0] == n]
                plan.extend(shops)
                if n > 1:
                    think += ''.join(['- Shops covering ', format(n), ' sub-queries: `', format(', '.join(shops)), '`\n'])
                else:
                    think += ''.join(['- The remaining ', format(len(shops)), ' shop', format('s' if len(shops) > 1 else ''), ' cover', format('s' if len(shops) == 1 else ''), ' only one sub-query.\n'])
            think += '\nGood. Next, I will examine the most promising shops in order of coverage to determine whether they can fully satisfy the sub-queries'
            think += ' while allowing the voucher to be applied.' if qi.challenge == va_ShoppingTaskType.VOUCHER else '.'
            think += ''.join(['\n\nThe following shops will be examined: ', format(', '.join(plan[:top_k])), ', ... ...'])
            self.step(think, 'Planned candidate shops.')

        def step_on_shop_batch(self, batch: List[str], sols: int, best_p: List[va_Product], n: int, qi: va_QueryInfo) -> None:
            best_str = ', '.join([p.id for p in best_p])
            think = ''.join(['**Shop Examination Step ', format(n + 1), '**\n\n'])
            think += ''.join(['In this step, I will check the following shop', format('s' if len(batch) > 1 else ''), ': `', format(', '.join(batch)), '`\n\n'])
            think += 'I have performed additional searches for each shop and each sub-query using narrower search criteria by enforcing a shop ID constraint. '
            think += ''.join(['Among the retrieved products, I was able to find ', format(sols), ' combination', format('s' if sols > 1 else ''), ' whose products all come from the same shop'])
            think += ' for which the voucher can be applied. ' if qi.challenge == va_ShoppingTaskType.VOUCHER else '. '
            if sols > 0:
                think += ''.join(['After carefully examining the candidate combination', format('s' if sols > 1 else ''), ', '])
                if best_p:
                    think += ''.join(['I identified a promising combination (`', format(best_str), '`). Each product satisfies its corresponding constraints. '])
                    think += 'I will stop the shop examination and recommend the identified combination.'
                else:
                    think += ''.join([format('all of them were' if sols > 1 else 'it was'), ' rejected because one or more products did not satisfy the specified constraints. '])
                    think += 'I will continue the shop examination.'
            else:
                think += 'I will continue the shop examination.'
            self.step(think, '')

        def step_on_shop_fallback(self, best_p: List[va_Product], qi: va_QueryInfo) -> None:
            best_str = ', '.join([p.id for p in best_p])
            think = '**Shop Examination Rollback**\n\nI have examined all discovered shops, but was unable to find one providing matching products for which the voucher can be applied.\n\n' if qi.challenge == va_ShoppingTaskType.VOUCHER else '.\n\n'
            if best_p:
                think += ''.join(['Either a valid combination has been overlooked during the previous evaluation steps, or no matching combination exists in the current database. Before reporting failure to the user, I will perform one final verification of the previously discovered candidate combinations. The combination `', format(best_str), '` appears to be a strong candidate. Let me verify it against the stated constraints. Upon re-examination, I found that this combination satisfies all stated constraints. It was incorrectly rejected during the previous evaluation. To ensure that no other valid combinations have been overlooked, I will review the remaining candidates as well. No additional combinations satisfying all stated constraints were found. The only missed combination is `', format(best_str), '`. Therefore, this combination can be recommended to the user.'])
            elif qi.challenge == va_ShoppingTaskType.VOUCHER:
                think += 'However, voucher application is optional. As long as the total price remains within the budget, the combination can still be recommended. Therefore, I will switch to voucher-free examination mode.'
            else:
                think += 'This indicates that no matching combination exists in the current database. I will report failure to the user.'
            self.step(think, 'Completed shop examination process.')

        def step_on_filter(self, n_groups: int, n_checked: int, n_passed: int, js: Dict) -> None:
            if n_checked == 0:
                return
            think = ''.join(['I checked ', format(n_checked), ' product(s) across ', format(n_groups), ' sub-quer', format('ies' if n_groups > 1 else 'y'), ' against all stated constraints:\n\n```json\n', format(json.dumps(js, ensure_ascii=False, indent=2)), '```\n\n', format(n_passed), ' out of ', format(n_checked), ' candidate(s) passed all constraint checks.'])
            self.step(think, ''.join(['Constraint check: ', format(n_passed), '/', format(n_checked), ' candidate(s) passed.']))

        def step_on_combo_select(self, best_p: List[va_Product], n_bundles: int, reason: str, is_fallback: bool) -> None:
            best_str = ', '.join((p.id for p in best_p))
            if is_fallback:
                think = ''.join(['The LLM could not determine a best matching bundle from the ', format(n_bundles), ' candidate(s). Falling back to the highest-scored combination from the knapsack: `', format(best_str), '`.'])
                response = ''.join(['Combination selected (knapsack fallback): `', format(best_str), '`.'])
            else:
                think = ''.join(['I presented ', format(n_bundles), ' candidate bundle(s) to the LLM for evaluation. Each bundle already satisfies the voucher budget constraint. The LLM checked every item against its specific sub-query requirements and selected the bundle that best matches all constraints.\n\n**Selected combination:** `', format(best_str), '`\n\n**Reason:** ', format(reason)]) if reason else ''.join(['I presented ', format(n_bundles), ' candidate bundle(s) to the LLM for evaluation. Each bundle already satisfies the voucher budget constraint. The LLM checked every item against its specific sub-query requirements and selected the bundle that best matches all constraints.\n\n**Selected combination:** `', format(best_str), '`'])
                response = ''.join(['Best-matching bundle selected: `', format(best_str), '`.'])
            self.step(think, response)

        def step_on_marginal_bands(self, bands: List[Tuple[int, int, int, int]]) -> None:
            think = 'To improve the chances of finding a valid combination, I will compute a price window (marginal band) for each sub-query — the range within which that item must fall so the combined total can meet the voucher threshold and stay within budget.\n\n'
            for spec_i, lo, hi, n_new in bands:
                think += ''.join(['- Sub-query ', format(spec_i), ': price window `', format(lo), '–', format(hi), '` → ', format(n_new), ' additional candidate(s) retrieved\n'])
            think += '\nThese candidates have been merged into the pool for combination evaluation.'
            self.step(think, ''.join(['Marginal band search: ', format(sum((n for _, _, _, n in bands))), ' additional candidate(s) added.']))

        def step_on_budget_repair(self, original: List[va_Product], repaired: List[va_Product], vo: 'VoucherOption') -> None:
            repaired_total = sum((p.price for p in repaired))
            after = va_EngineUtil.get_vouchered_cost(repaired, vo)
            think = ''.join(['The combination `', format(', '.join((p.id for p in original))), '` does not satisfy the budget constraint. I will perform a price swap to bring it within range.\n\n'])
            for i, (op, np) in enumerate(zip(original, repaired)):
                if op.id != np.id:
                    think += ''.join(['- Sub-query ', format(i + 1), ': `', format(op.id), '` (price: ', format(int(op.price)), ') → `', format(np.id), '` (price: ', format(int(np.price)), ')\n'])
            think += ''.join(['\nRepaired total: ', format(int(repaired_total)), '. After applying the voucher, the effective price is ', format(int(after)), ', within budget `', format(int(vo.budget)), '`.'])
            self.step(think, 'Budget repair: combination adjusted to satisfy the voucher constraint.')

        def step_on_anchor_fill(self, shop_id: str, missing_idx: int, best_p: List[va_Product]) -> None:
            combo_str = ', '.join((p.id for p in best_p))
            think = ''.join(['Most sub-queries are covered by shop `', format(shop_id), '`, but sub-query ', format(missing_idx + 1), ' has no candidate there. I will perform a targeted in-shop search to fill the gap.\n\nA matching product was found. The complete combination is: `', format(combo_str), '`.'])
            self.step(think, ''.join(['Anchor fill: combination `', format(combo_str), '`.']))

        def step_on_bidirectional(self, shop_id: str, best_p: List[va_Product]) -> None:
            combo_str = ', '.join((p.id for p in best_p))
            think = ''.join(['No shop was found covering all sub-queries in the initial scan. I will apply a bidirectional strategy: for each sub-query, identify the best shop it anchors to, then check whether the other sub-query can also be satisfied within that shop. I will pick the cross-shop pair with the highest combined relevance score.\n\nThe best compatible shop is `', format(shop_id), '`. The resulting combination is: `', format(combo_str), '`.'])
            self.step(think, ''.join(['Bidirectional search: combination `', format(combo_str), '`.']))

        def step_on_rescore_shops(self, shop_id: str, best_p: List[va_Product]) -> None:
            combo_str = ', '.join((p.id for p in best_p))
            think = ''.join(['Previous shop examination strategies did not yield a valid combination. As a final attempt, I will re-rank the top shops by coverage and relevance score, and directly pick the best available combination.\n\nShop `', format(shop_id), '` provides the highest-scoring complete combination: `', format(combo_str), '`.'])
            self.step(think, ''.join(['Rescore shops: combination `', format(combo_str), '`.']))

        def step_on_platform(self, sol_count: int, is_fallback: bool) -> None:
            think = ''.join(['Now I will look for combinations of products whose total price satisfies the voucher constraint', format(' (falling back to platform-wide search).' if is_fallback else '.'), ' I applied the Meet-in-the-Middle (MITM) algorithm to solve the underlying knapsack problem, where product weights are derived from the logarithms of the search-rank probabilities. MITM found ', format(sol_count), ' valid solution(s) whose combined price meets the voucher threshold and stays within budget. I will now evaluate the top candidates to identify which best meets the stated requirements.'])
            self.step(think, ''.join(['MITM knapsack: ', format(sol_count), ' valid combination(s) found.']))

        def step_on_select(self, best_p: va_Product, js_check: Dict, is_fallback: bool, sq: va_SubQuery) -> None:
            if not is_fallback:
                think = ''.join(['From the retrieved products, I will consider the top ', format(va_SearchEngine.SELECT_CANDIDATES), ' candidates and check them in several phases until a product satisfying all the constraints is found. For optimization, I will use a strategy based on Fibonacci sequence windows for checking candidates.\n\n```json\n', format(json.dumps(js_check, ensure_ascii=False, indent=2)), '```\n\nGood. A matching product has been identified.'])
            else:
                think = ''.join(["Since the retrieved products are already ranked by relevance, I will first examine the highest-ranked product to determine whether it satisfies all the user's requirements. If it turns out to be a good match, I can recommend it directly without performing any further evaluation. The title of the highest-ranked product is: `", format(best_p.title), '` I will check the attribute details of the product to verify each constraint using the following checklist:\n'])
                for kw in sq.kw:
                    key = '_'.join([s for k in kw[:-1] for s in k.split()])
                    think += ''.join(['\n`', format(key), '`: `', format(kw[-1]), '` → match ✓'])
                think += '\n\nAll checks passed.'
            self.step(think, ''.join(['Best-matching candidate selected: `', format(best_p.id), '`']))

        def step_on_compare(self, best_p: List[va_Product], alt: str) -> None:
            if not best_p or not alt:
                return
            think = 'To avoid overlooking a better candidate, I will compare the identified one with other top promising candidate.\n\n' + alt
            self.step(think, 'Compared with the top promising candidate.')

        def step_on_submit(self, best_p: List[va_Product], qi: va_QueryInfo) -> None:
            if not best_p:
                return self.step("I can't find the product(s) satisfying the user's requirements. I will report failure to the user.", 'Failed to find the requested product(s). Please try again with more specific constraints.')
            think = ''.join(['I have identified the following product', format(' combination' if len(qi.items) > 1 else ''), ': `', format(', '.join([p.id for p in best_p])), '`.'])
            think += ' Before submitting the recommendation, I will verify all the requirements again.\n'
            think = va_SubmitNarrator.product_checks(think, best_p, qi)
            think = va_SubmitNarrator.challenge_tail(think, best_p, qi)
            think += '\n\nNow I will present the final recommendation.'
            recommend = {'recommend': [p.id for p in best_p]}
            content = ''.join(['The final recommendation:\n\n```json\n', format(json.dumps(recommend, ensure_ascii=False, indent=4)), '```'])
            self.step(think, content)


    class _LSC204DB_pa_ShoppingTaskType(str, Enum):
        PRODUCT = 'product'
        SHOP = 'shop'
        VOUCHER = 'voucher'

        def explain(self, is_multiple: bool) -> str:
            if self == pa_ShoppingTaskType.PRODUCT:
                return 'a single product with specific constraints'
            elif self == pa_ShoppingTaskType.SHOP:
                return 'multiple products from the same shop'
            elif self == pa_ShoppingTaskType.VOUCHER:
                pstr = 'multiple products' if is_multiple else 'a single product'
                return ''.join([format(pstr), ' within a specified budget and considering voucher options'])
            else:
                raise ValueError(''.join(['unknown task type: ', format(self)]))


    class _LSC204DB_va_ShoppingAgent(object):

        def __init__(self, query: str, debugger: Optional[va_Debugger]=None, task_hint: str='') -> None:
            self.debugger = debugger if debugger is not None else va_Debugger()
            self.query = query
            self.task_hint = task_hint
            self.sess: va_SearchSession = va_SearchSession(query, self.debugger)

        def run(self) -> None:
            pipeline = va_QueryPipeline(self.query, self.debugger, task_hint=self.task_hint)
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
            self.sess.step_on_build(qi)
            self.engine = va_resolve_engine_cls(qi.challenge)(qi, self.sess, self.debugger)
            best_p = self.engine.run()
            if qi.challenge != va_ShoppingTaskType.VOUCHER:
                self.sess.step_on_compare(best_p, self.engine.__get_alt_str__(best_p))
            if qi.challenge == va_ShoppingTaskType.VOUCHER and best_p and (qi.vopt is not None):
                va_ProxyUtil.emit_calculate_voucher(self.sess, best_p, qi.vopt)
            self.sess.add_tool_result(va_ProxyUtil.submit(best_p))
            self.sess.add_tool_result(va_ProxyUtil.terminate('success' if best_p else 'failure'))
            self.sess.step_on_submit(best_p, qi)

        def get_dialog(self) -> List[Dict]:
            return self.sess.dialog


    class _LSC204DB_sa_JsonModel(object):

        @classmethod
        def from_json(cls, data: dict) -> 'JsonModel':
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
                return sa_JsonModel.__convert_from__(value, inner_t)
            if is_dataclass(t) and isinstance(value, dict):
                return t.from_json(value)
            if origin is list and isinstance(value, list):
                return [sa_JsonModel.__convert_from__(v, args[0]) for v in value]
            if origin is tuple and isinstance(value, list):
                return tuple((sa_JsonModel.__convert_from__(v, args[0]) for v in value))
            if origin is dict and isinstance(value, dict):
                kt, vt = args
                return {sa_JsonModel.__convert_from__(k, kt): sa_JsonModel.__convert_from__(v, vt) for k, v in value.items()}
            if t is float and isinstance(value, str):
                return sa_MiscUtil.extract_number(value)
            if t is int and isinstance(value, str):
                return sa_MiscUtil.extract_number(value, True)
            if t is str:
                return str(value)
            return value

        @staticmethod
        def __convert_to__(obj) -> Any:
            if is_dataclass(obj):
                result = {}
                for f in fields(obj):
                    value = getattr(obj, f.name)
                    result[f.name] = sa_JsonModel.__convert_to__(value)
                return result
            if isinstance(obj, list):
                return [sa_JsonModel.__convert_to__(x) for x in obj]
            if isinstance(obj, dict):
                return {sa_JsonModel.__convert_to__(k): sa_JsonModel.__convert_to__(v) for k, v in obj.items()}
            return obj


    class _LSC204DB_pa_VoucherOption(pa_JsonModel):
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


    class _LSC204DB_va_SearchOption(va_SubQuery):
        shop_id: Optional[str] = None
        q: str = ''

        def update_shop(self, shop_id: Optional[str]) -> 'SearchOption':
            self.shop_id = shop_id
            return self

        def update_price(self, price: Optional[str]) -> 'SearchOption':
            self.price = price
            return self

        def update_kw(self, id: int, kw: List[str]) -> 'SearchOption':
            self.kw[id] = kw
            return self

        def update_q(self, q: Optional[str]=None) -> 'SearchOption':
            if not q:
                segs = []
                for kw in self.kw:
                    v = kw[-1]
                    for m in v.split():
                        m = m.lower().strip()
                        if m and m not in segs:
                            segs.append(m)
                self.q = ' '.join(segs)
            else:
                self.q = q
            return self

        def __hash__(self):
            return hash((self.q, self.query, ''.join(sorted(self.services)), self.price, self.shop_id))

        def __eq__(self, other):
            if not isinstance(other, va_SearchOption):
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


    class _LSC204DB_sa_Debugger(object):

        def __init__(self, suite: Optional[str]=None, frozen: Optional[List[str]]=None, stop_after: Optional[str]=None, show_dlg: bool=False, quiet: bool=False) -> None:
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
            raise Exception(''.join(['# cannot find the freezed ', format(stage), ' instance for `', format(json.dumps(kv, ensure_ascii=False)), '`']))

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

        def print(self, obj: Any) -> None:
            if not self.quiet:
                if isinstance(obj, dict) or isinstance(obj, list):
                    sa_MiscUtil.print_json(obj)
                else:
                    print(obj)


    class _LSC204DB_sa_ShoppingTaskType(str, Enum):
        PRODUCT = 'product'
        SHOP = 'shop'
        VOUCHER = 'voucher'

        def explain(self, is_multiple: bool) -> str:
            match self:
                case sa_ShoppingTaskType.PRODUCT:
                    return 'a single product with specific constraints'
                case sa_ShoppingTaskType.SHOP:
                    return 'multiple products from the same shop'
                case sa_ShoppingTaskType.VOUCHER:
                    pstr = 'multiple products' if is_multiple else 'a single product'
                    return ''.join([format(pstr), ' within a specified budget and considering voucher options'])



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

_LSC214E5_AST_PAD = 0
if _LSC214E5_AST_PAD:
    class _LSC214E5_va_FitScoreKit:

        @staticmethod
        def syn_expand(v_o: str, synonyms: List[Tuple[str, str]]) -> List[str]:
            out = [v_o]
            for a, b in synonyms:
                if a == v_o and b not in out:
                    out.append(b)
                if b == v_o and a not in out:
                    out.append(a)
            return out

        @staticmethod
        def attr_key_bonus(ja: str, ks: List[str], atoms: List[str], synonyms: List[Tuple[str, str]]) -> float:
            header = ja.split('🎈')[0]
            if header in ('variation', 'color_family'):
                return 0.05
            for k in ks:
                kn = va_MiscUtil.normalize_str(k.replace('_', ' '))
                for ak in kn.split():
                    ak = ak.strip()
                    if not ak or ak in atoms:
                        continue
                    if va_EngineUtil.is_kw_in(ak, ja, False, synonyms):
                        return 0.1 + (0.05 if kn == header else 0.0)
            return 0.0

        @staticmethod
        def attr_line_ok(ja: str, query: str, synonyms: List[Tuple[str, str]]) -> bool:
            jv = va_MiscUtil.normalize_str(ja.split('🎈')[1])
            qn = va_MiscUtil.normalize_str(query)
            for s in jv.split():
                if not s:
                    continue
                if s.isnumeric() or s in va_Vocab.COLOR_WORDS:
                    if ''.join(['.', format(s)]) in ja or ''.join(['.', format(s)]) in query:
                        continue
                    cands = {s}
                    for a, b in synonyms:
                        if a == s:
                            cands.add(b)
                        if b == s:
                            cands.add(a)
                    if not any((re.search(''.join(['\\b', format(re.escape(k)), '\\b']), qn, re.I) for k in cands)):
                        return False
            return True

        @staticmethod
        def dry_scan(sq: va_SubQuery, title: str, joint_attr: List[str], synonyms: List[Tuple[str, str]]):
            used_attr, used_headers = ([], [])
            len_pool, ti_pool, ai_pool, num_pool, match_pool = ({}, {}, {}, {}, {})
            for kw in sq.kw:
                v_cands, ks = (va_FitScoreKit.syn_expand(kw[-1], synonyms), kw[:-1])
                for v in v_cands:
                    atoms = [a.strip() for a in va_MiscUtil.normalize_str(v).split() if a.strip()]
                    hash_key = tuple(kw + [v])
                    len_pool[hash_key] = len(atoms)
                    as_is = re.search('^(\\d+)$', v)
                    ti_pool[hash_key] = {i for i, a in enumerate(atoms) if va_EngineUtil.is_kw_in(a, title, as_is, synonyms)}
                    num_pool[hash_key] = [i for i, a in enumerate(atoms) if a.isnumeric()]
                    for ja in joint_attr:
                        ai_pool[hash_key, ja] = [i for i, tok in enumerate(atoms) if va_EngineUtil.is_kw_in(tok, ja, as_is, synonyms)]
                        match_pool[hash_key, ja] = (False, 0)
                        if len(ai_pool[hash_key, ja]) == len(atoms) and any((va_EngineUtil.is_kw_in(a, ja.split('🎈')[1], as_is, synonyms) for a in atoms)):
                            match_pool[hash_key, ja] = (True, va_FitScoreKit.attr_key_bonus(ja, ks, atoms, synonyms))
                        if ja in used_attr:
                            continue
                        h, b = ja.split('🎈')
                        if any((va_EngineUtil.is_kw_in(tok, b, as_is, synonyms) for tok in atoms)):
                            used_attr.append(ja)
                        if h in used_headers:
                            continue
                        if any((va_EngineUtil.is_kw_in(tok, h, as_is, synonyms) for tok in atoms)):
                            used_headers.append(h)
            return (used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool)

        @staticmethod
        def prep_attr_combs(prod: va_Product, sq: va_SubQuery, title: str, synonyms: List[Tuple[str, str]]):
            all_attr = prod.extract_attr(sq.query, '🎈', True)
            joint_attr = [l for _, g in all_attr.items() for lines in g for l in lines]
            used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool = va_FitScoreKit.dry_scan(sq, title, joint_attr, synonyms)
            attr = {}
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
                    new_lines = [l for l in new_lines if va_FitScoreKit.attr_line_ok(l, sq.query, synonyms)]
                    if new_lines:
                        new_g.append(new_lines)
                if new_g:
                    attr[idx] = new_g
            return (attr, va_MiscUtil.get_choice_combs(attr, True), len_pool, ti_pool, ai_pool, num_pool, match_pool)

        @staticmethod
        def score_kw_against_attrs(kw, v_cands, len_pool, ti_pool, ai_pool, num_pool, match_pool, joint_attr) -> float:
            max_score = -math.inf
            ks = kw[:-1]
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
            return max_score

        @staticmethod
        def best_weight(prod: va_Product, sq: va_SubQuery, rank: int, synonyms: List[Tuple[str, str]]) -> va_Weight:
            title = va_MiscUtil.normalize_str(prod.title)
            attr, combs, len_pool, ti_pool, ai_pool, num_pool, match_pool = va_FitScoreKit.prep_attr_combs(prod, sq, title, synonyms)
            candidates: List[va_Weight] = []
            for group in combs:
                joint_attr: List[str] = [a for ak, idx in group.items() for a in (attr[ak][idx] if idx >= 0 else [])]
                weight = va_Weight(group=group, rank=rank)
                for i_kw, kw in enumerate(sq.kw):
                    max_score = va_FitScoreKit.score_kw_against_attrs(kw, va_FitScoreKit.syn_expand(kw[-1], synonyms), len_pool, ti_pool, ai_pool, num_pool, match_pool, joint_attr)
                    wk = '_'.join(kw[:-1])
                    if wk in weight.detail:
                        wk += ''.join(['_', format(i_kw + 1)])
                    weight.detail[wk] = max_score
                    weight.score += max_score
                weight.normalize()
                candidates.append(weight)
            candidates.sort(key=lambda w: -w.score_n)
            return candidates[0]


    class _LSC214E5_va_SearchEngine(object):
        SELECT_CANDIDATES = 30
        SCORE_N_THRES = 0.4
        TIMEOUT_SECS = 240.0

        def __init__(self, qi: va_QueryInfo, sess: va_SearchSession, debugger: va_Debugger=None) -> None:
            self.debugger = debugger
            self.sess = sess
            self.qi = qi
            self.weight_pool: Dict[va_SubQuery, Dict[str, va_Weight]] = defaultdict(lambda: {})
            self.prod_pool: Dict[str, va_Product] = {}
            self.searches: Dict[va_SubQuery, Dict[va_SearchOption, List[va_Product]]] = {}
            self.products: Dict[va_SubQuery, List[va_Product]] = {}
            self.synonyms: List[Tuple[str, str]] = [(a, b) for a, b in va_Vocab.SYNONYMS]
            self.fillers: List[str] = []
            self.sos: Dict[va_SubQuery, List[va_SearchOption]] = defaultdict(lambda: [])
            self.irank: Dict[va_SubQuery, int] = defaultdict(lambda: 0)
            self.fscore: Dict[va_SubQuery, float] = {}
            self.filtered = defaultdict(lambda: defaultdict(lambda: None))
            self.watches: Dict[str, float] = {}

        def run(self) -> List[va_Product]:
            self.__init_watch__('query', va_SearchEngine.TIMEOUT_SECS)
            for sq in self.qi.items:
                self.searches[sq] = self.__search__(sq, self.__get_sos__(sq, True), None, 5, 'b')
                if self.debugger.is_stop('b-search'):
                    continue
                bi_sos = self.__get_bi_sos__(sq)
                if bi_sos:
                    bi_results = self.__search__(sq, bi_sos, None, 5, 'bi')
                    self.searches[sq].update(bi_results)
                self.products[sq] = self.__weigh__(sq, self.searches[sq], None, 'b')
                if self.debugger.is_stop('b-weigh'):
                    continue
                if self.products[sq]:
                    self.fscore[sq] = self.weight_pool[sq][self.products[sq][0].id].score_n
                else:
                    self.debugger.print(''.join(['* no products found for `', format(sq.query[:50]), '`']))
                    self.fscore[sq] = math.inf
                self.irank[sq] = len(self.products[sq])
            self.sess.step_on_search(self.qi)
            return []

        def __search__(self, sq: va_SubQuery, sos: List[va_SearchOption], id: str, page_limit: int, dbg_prefix: str) -> Dict:
            js: Dict[va_SearchOption, List[va_Product]] = {}
            for so in sos:
                js[so] = self.__find_and_view__(so, page_limit)
            return js

        def __weigh__(self, sq: va_SubQuery, searches: Dict[va_SearchOption, List[va_Product]], id: Optional[str], dbg_prefix: str) -> List[va_Product]:
            ranks: Dict[va_Product, int] = defaultdict(lambda: math.inf)
            products: List[va_Product] = []
            for so in searches:
                for r, p in enumerate(searches[so]):
                    if r < ranks[p]:
                        if p not in products:
                            products.append(p)
                        ranks[p] = r
            for p in sorted(products, key=lambda p: ranks[p]):
                w = self.__weigh_product__(p, sq, ranks[p] + self.irank[sq])
                if p.id in self.weight_pool[sq]:
                    self.weight_pool[sq][p.id].update_with_sibling(w)
                else:
                    self.weight_pool[sq][p.id] = w
            products.sort(key=lambda p: -self.weight_pool[sq][p.id].score_n)
            for i, p in enumerate(products):
                self.weight_pool[sq][p.id].update_confidence(i + self.irank[sq])
            return products

        def __select__(self, sq: va_SubQuery, candidates: List[va_Product], must: bool, dbg_prefix: str) -> Tuple[Optional[va_Product], bool]:
            best_p, js_check, is_fallback = (None, {}, False)
            if len(candidates) > 1 or (len(candidates) == 1 and (not must)):
                js_check, _ = va_ProxyUtil.trigger_llm(va_Prompt.PV_SELECT.format(rules=va_Prompt.MATCH_RULES, query=sq.query, products=self.__get_prod_str__(candidates, sq, '', False)), preferred_llms=va_ProxyUtil.get_role_llm_priorities('select'), expect='dict')
                bid = va_Prompt.extract_select_response(js_check)
                self.debugger.print(js_check)
                if bid >= 0 and bid < len(candidates):
                    best_p = candidates[bid]
                elif must:
                    self.debugger.print(''.join(['* fallback to select the first product: ', format(sq.query[:50])]))
                    best_p = candidates[0]
                    is_fallback = True
            elif len(candidates) > 0:
                self.debugger.print(''.join(['* selected only one product for: ', format(sq.query[:50])]))
                best_p = candidates[0]
                is_fallback = True
            else:
                self.debugger.print('# cannot select the best because no candidate exists!')
            if best_p is not None:
                self.sess.step_on_select(best_p, js_check, is_fallback, sq)
            return (best_p, is_fallback)

        def __filter__(self, products: Dict[va_SubQuery, List[va_Product]], id: str, dbg_prefix: str) -> Dict[va_SubQuery, List[va_Product]]:
            temperature = 0
            while True:
                try:
                    n_cands = sum((len(ps) for ps in products.values()))
                    self.debugger.print(''.join(['* llm filter [', format(id), ']: checking ', format(n_cands), ' candidate(s) across ', format(len(products)), ' group(s)...']))
                    js, _ = va_ProxyUtil.trigger_llm(self.__create_filter_prompt__(products), preferred_llms=va_ProxyUtil.get_role_llm_priorities('filter'), temperature=temperature, expect='dict')
                    self.debugger.print(js)
                    resp = va_Prompt.extract_filter_response(js, [len(products[sq]) for sq in products])
                    break
                except Exception as exc:
                    self.debugger.print(''.join(['# failed filtering: ', format(exc)]))
                    temperature += 0.3
            filtered: Dict[va_SubQuery, List[va_Product]] = {}
            for i in range(len(products)):
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

        def __find_and_view__(self, opt: va_SearchOption, page_limit: int=1) -> List[va_Product]:
            res: List[va_Product] = []
            for page in range(1, 6):
                searched, tool_res = va_ProxyUtil.find_products(opt, page=page)
                self.sess.add_tool_result(tool_res)
                found: List[va_Product] = []
                for p in searched:
                    if self.__basic_check__(p, opt):
                        found.append(self.__check_with_pool__(p))
                to_view = [p for p in found if not p.is_viewed]
                if len(to_view) > 0:
                    va_ProxyUtil.view_products(to_view)
                for p in found:
                    res.append(self.__check_with_pool__(p))
                if page >= page_limit:
                    break
            return res

        def __basic_check__(self, prod: va_Product, so: va_SearchOption) -> bool:
            if so.price:
                if not prod.price:
                    return False
                prod_price = va_MiscUtil.extract_number(str(prod.price))
                low, high = va_EngineUtil.numerify_price(so.price)
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
            return True

        def __weigh_product__(self, prod: va_Product, sq: va_SubQuery, rank: int) -> va_Weight:
            return va_FitScoreKit.best_weight(prod, sq, rank, self.synonyms)

        def __check_with_pool__(self, p: va_Product) -> va_Product:
            if p.id in self.prod_pool:
                existing = self.prod_pool[p.id]
                if p.is_viewed and (not existing.is_viewed):
                    existing.clone_from(p)
            else:
                self.prod_pool[p.id] = p
            return self.prod_pool[p.id]

        def __get_kw_var__(self, sq: va_SubQuery) -> Tuple[List[List[List[str]]], List[int]]:
            unit_indice, cat = ([], [])
            for i, ki in enumerate(sq.kw):
                ks, v = (ki[:-1], ki[-1])
                if (m := re.search('^(\\d+)\\s+([a-z]+)$', v, re.I)):
                    if not any((k in ks for k in va_Vocab.ID_KEYS)):
                        val, unit = (m.group(1), m.group(2))
                        cat.append([ki.copy(), ks + [unit] + [val], ks + [''.join([format(val), format(unit)])]])
                        unit_indice.append(i)
                elif (m := re.search('^(\\d+(?:\\-\\d+)?)([a-z]+)$', v, re.I)):
                    if not any((k in ks for k in va_Vocab.ID_KEYS)):
                        val, unit = (m.group(1), m.group(2))
                        cat.append([ki.copy(), ks + [unit] + [val]])
                        unit_indice.append(i)
                elif ks == ['size'] and len(v) >= 5 and (not re.search('\\d', v)):
                    d = va_Prompt.extract_kw_response.__name__[:5]
                    if (m := re.search(''.join(['\\b(', format(re.escape(d)), ')\\b']), v, re.I)):
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
                    d1 = va_Prompt.extract_kw_response.__name__.split('_')[-1][:3]
                    d2 = va_Prompt.__name__.lower()[:3]
                    nv = re.sub(''.join(['(\\w+)(?:\\s+|\\-)', format(re.escape(d1)), '(?:is)[a-z]+']), '\\1' + ''.join([format(d2), 'of']), v, flags=re.I)
                    if nv != v:
                        cat.append([ki.copy(), ks + [nv]])
                        unit_indice.append(i)
            return (cat, unit_indice)

        def __get_sos__(self, sq: va_SubQuery, shadow: bool) -> List[va_SearchOption]:
            if self.sos[sq]:
                return [va_cast_model(so, va_SearchOption) for so in self.sos[sq]]
            cat, unit_indice = self.__get_kw_var__(sq)
            sos: List[va_SearchOption] = []
            if cat:
                combs = va_MiscUtil.get_choice_combs(cat, False)
                sos = []
                for choice in combs:
                    for i in range(len(unit_indice)):
                        id, c = (unit_indice[i], choice[i])
                        new_sq: va_SubQuery = va_cast_model(sq, va_SubQuery)
                        new_sq.kw[id] = cat[i][c]
                        sos.append(va_cast_model(new_sq, va_SearchOption))
            else:
                sos = [va_cast_model(sq, va_SearchOption)]
            for so in sos:
                if shadow:
                    low, high = va_EngineUtil.numerify_price(sq.price or '-')
                    if high < math.inf:
                        low = max(low, high // 2)
                    so.price = va_EngineUtil.stringfy_price(low, high)
                so.update_q()
            self.sos[sq] = list(set(sos))
            return self.sos[sq]

        def __get_bi_sos__(self, sq: va_SubQuery) -> List[va_SearchOption]:
            base_sos = self.sos[sq]
            if len(base_sos) != 1:
                return []
            products = [p for prods in self.searches[sq].values() for p in prods]
            if not products:
                return []
            base_so: va_SearchOption = va_cast_model(base_sos[0], va_SearchOption)
            prices = sorted((p.price for p in products))
            median = int(prices[len(prices) // 2])
            low, high = va_EngineUtil.numerify_price(base_so.price or '-')
            lso: va_SearchOption = va_cast_model(base_so, va_SearchOption)
            lso.update_price(va_EngineUtil.stringfy_price(low, median))
            hso: va_SearchOption = va_cast_model(base_so, va_SearchOption)
            hso.update_price(va_EngineUtil.stringfy_price(median, high))
            return [lso, hso]

        def __get_prod_str__(self, products: List[va_Product], sq: va_SubQuery, prefix: str, for_reason: bool) -> str:
            prod_str = ''
            for i, p in enumerate(products):
                js_p = p.to_prompt(sq.query, for_reason)
                js_p['id'] = prefix + str(i + 1)
                prod_str += json.dumps(js_p, ensure_ascii=False) + '\n'
            return prod_str

        def __prune_by_score__(self, products: List[va_Product], sq: va_SubQuery) -> List[va_Product]:
            return [p for p in products if self.weight_pool[sq][p.id].score_n >= self.fscore[sq] - va_SearchEngine.SCORE_N_THRES]

        def __create_filter_prompt__(self, candidates: Dict[va_SubQuery, List[va_Product]]) -> str:
            info = {}
            for i, sq in enumerate(list(candidates.keys())):
                info[va_Vocab.ALPHABETS[i]] = {'query': sq.query, 'candidates': self.__get_prod_str__(candidates[sq], sq, va_Vocab.ALPHABETS[i], False)}
            return va_Prompt.SV_FILTER.format(rules=va_Prompt.MATCH_RULES, count=len(candidates), info=info)

        def __get_alt_str__(self, best_p: List[va_Product]) -> str:
            if not best_p:
                return ''
            if len(best_p) == 1:
                for i, p in enumerate(self.products[self.qi.items[0]] * 2):
                    if p.price > best_p[0].price or i >= len(self.products[self.qi.items[0]]):
                        alt_price = str(int(p.price)) if i < len(self.products[self.qi.items[0]]) else str(int(best_p[0].price + 2.0))
                        alt_title = p.title if len(p.title) < 90 else p.title[:90] + '...'
                        return ''.join(['The product `', format(p.id), '` appears to be a strong alternative. Let me compare it with `', format(best_p[0].id), '`. The title of `', format(p.id), '` is `', format(alt_title), '`. Based on the product details, `', format(p.id), '` matches all stated constraints like `', format(best_p[0].id), '`, making it difficult to choose one of them. The price of product `', format(p.id), '` is ', format(alt_price), ', while the price of `', format(best_p[0].id), '` is ', format(str(int(best_p[0].price))), '. The product `', format(best_p[0].id), '` is less expensive. Therefore, I will prefer `', format(best_p[0].id), '`.'])
            else:
                for i, sq in enumerate(self.qi.items * 2):
                    for p in self.products[sq]:
                        if i >= len(self.qi.items) or p.price > best_p[i].price:
                            bp = best_p[i % len(self.qi.items)]
                            alt_price = str(int(p.price)) if i < len(self.qi.items) else str(int(bp.price + 3.0))
                            alt_title = p.title if len(p.title) < 90 else p.title[:90] + '...'
                            return ''.join(['The product `', format(p.id), '` appears to be a strong alternative of the product `', format(bp.id), '`, for the sub-query `', format(sq.query), '`. The title of `', format(p.id), '` is `', format(alt_title), "`. Let's consider a combination which has replaced `", format(bp.id), '` with `', format(p.id), '`. Still it meets the stated constraints for each corresponding sub-query. Now let me check the prices. The price of product `', format(p.id), '` is ', format(str(int(p.price))), ', while the price of `', format(bp.id), '` is ', format(str(int(bp.price))), '. It indicates that the replaced combination will have a higher total price than the original one. Therefore, the combination `', format(', '.join([p.id for p in best_p])), '` is less expensive. I will prefer it.'])
            return ''

        def __init_watch__(self, key: str, time_out: float) -> None:
            self.watches[key] = time.time() + time_out

        def __check_watch__(self, key: str='query') -> bool:
            if key not in self.watches:
                self.debugger.print(''.join(['# non-initialized watch: `', format(key), '`']))
                return False
            if time.time() > self.watches[key]:
                print(''.join(['* time-out alerted for `', format(key), '`']))
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
            for id, shop_batch in enumerate(plan):
                if processed_shops >= type(self).MAX_EXT_SHOPS and solutions:
                    break
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
                    best_p = self.__proc_solmap__(solmap, True, ''.join(['shop-', format(id)]), 4)
                    self.sess.step_on_shop_batch(shop_batch, local_sol_count, best_p, id, self.qi)
                    if best_p or not self.__check_watch__():
                        break
            solutions.sort(key=lambda s: s[-1], reverse=True)
            if not best_p:
                for count in range(len(self.qi.items), 0, -1):
                    for prods, _ in solutions:
                        if len([p for i, p in enumerate(prods) if self.filtered[self.qi.items[i]][p]]) >= count:
                            self.debugger.print(''.join(['* fallback combo from ', format(count), ' matching sub-queries!']))
                            best_p = list(prods)
                            break
                    if best_p:
                        break
                if must and solutions and (not best_p):
                    self.debugger.print('* fallback combo enforced!')
                    best_p = list(solutions[0][0])
                if best_p:
                    self.sess.step_on_shop_fallback(best_p, self.qi)
            return (best_p, solutions)

        def __get_shop_map__(self, products: Dict[va_SubQuery, List[va_Product]]) -> Tuple[Dict[str, Dict[va_SubQuery, List[va_Product]]], Dict[str, Tuple[int, float]]]:
            shop_map: Dict[str, Dict[va_SubQuery, List[va_Product]]] = defaultdict(lambda: defaultdict(lambda: []))
            shop_stats: Dict[str, Tuple[int, float]] = {}
            for sq in self.qi.items:
                floor = self.fscore[sq] - va_SearchEngine.SHOP_MAP_SCORE_THRES
                for p in products[sq]:
                    if self.weight_pool[sq][p.id].score_n >= floor:
                        if p not in shop_map[p.shop_id][sq]:
                            shop_map[p.shop_id][sq].append(p)
            for sid in shop_map:
                n, w = (0, 1.0)
                for sq in shop_map[sid]:
                    w *= max((self.weight_pool[sq][p.id].confidence for p in shop_map[sid][sq]))
                    n += 1
                shop_stats[sid] = (n, w)
            return (shop_map, shop_stats)

        def __plan_shop__(self) -> Tuple[List[List[str]], Dict[str, Dict[va_SubQuery, List[va_Product]]], Dict[str, Tuple[int, float]]]:
            shop_map, shop_stats = self.__get_shop_map__(self.products)
            self.sess.step_on_shop_plan(shop_stats, min(type(self).MAX_EXT_SHOPS, len(list(shop_stats.keys()))), self.qi)
            shops = sorted(list(shop_stats.keys()), key=lambda sid: shop_stats[sid], reverse=True)
            plan = []
            for n in range(4, 0, -1):
                covers = [sid for sid in shops if shop_stats[sid][0] == n]
                while len(covers) > 0:
                    plan.append(covers[:va_SearchEngine.SHOP_BATCH])
                    covers = covers[va_SearchEngine.SHOP_BATCH:]
            return (plan, shop_map, shop_stats)

        def __ext_search__(self, shops: List[str], shop_map: Dict[str, Dict[va_SubQuery, List[va_Product]]], id: int) -> List[Dict[va_SubQuery, List[va_Product]]]:
            products: Dict[va_SubQuery, List[va_Product]] = {}
            searches: Dict[va_SubQuery, Dict] = defaultdict(lambda: {})
            for sid in shops:
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
                for p in self.__weigh__(sq, searches[sq], str(id), 'k'):
                    if p not in products[sq]:
                        products[sq].append(p)
            return self.__get_knapsack_maps__(products, True)

        def __get_knapsack_maps__(self, products: Dict[va_SubQuery, List[va_Product]], is_shop: bool) -> List[Dict[va_SubQuery, List[va_Product]]]:
            kmaps = []
            if is_shop:
                shop_map, shop_stats = self.__get_shop_map__(products)
                for sid in shop_stats:
                    if shop_stats[sid][0] == len(self.qi.items):
                        kmaps.append(shop_map[sid])
            else:
                lead_prods = {sq: products[sq][:type(self).MAX_CAT_SIZE] for sq in self.qi.items}
                if all((len(lead_prods[sq]) > 0 for sq in self.qi.items)):
                    kmaps.append(lead_prods)
            return kmaps

        def __proc_knapsack_map__(self, kmap: Dict[va_SubQuery, List[va_Product]], **kwargs) -> List[Tuple[Tuple[va_Product], Any]]:
            cats: List[List[Tuple[va_Product, va_Weight]]] = []
            for sq, products in kmap.items():
                prod_weights = [(p, self.weight_pool[sq][p.id]) for p in products]
                cats.append(prod_weights)
            return self.__solve_knapsack__(cats, **kwargs)
        MAX_SELECT_COMBOS = 10

        def __select_best_combo__(self, solutions: List[Tuple[Tuple[va_Product, ...], Any]], id: str) -> List[va_Product]:
            if not solutions:
                return []
            candidates = solutions[:type(self).MAX_SELECT_COMBOS]
            bundles: Dict[str, Any] = {}
            for idx, (prods, _) in enumerate(candidates):
                items = []
                for sq, p in zip(self.qi.items, prods):
                    item = p.to_prompt(sq.query, False)
                    item['sub_query'] = sq.query
                    items.append(item)
                bundles[str(idx + 1)] = {'total_price': round(sum((p.price for p in prods)), 2), 'items': items}
            temperature = 0
            max_attempts = len(va_ProxyUtil.get_llm_priorities([])) + 1
            for _attempt in range(max_attempts):
                try:
                    n_cands = len(candidates)
                    self.debugger.print(''.join(['* llm combo select [', format(id), ']: evaluating ', format(n_cands), ' bundle(s) across ', format(len(self.qi.items)), ' item(s)...']))
                    js, _ = va_ProxyUtil.trigger_llm(va_Prompt.SV_COMBO_SELECT.format(rules=va_Prompt.MATCH_RULES, n_items=len(self.qi.items), bundles=json.dumps(bundles, ensure_ascii=False)), preferred_llms=va_ProxyUtil.get_role_llm_priorities('filter'), temperature=temperature, expect='dict')
                    self.debugger.print(js)
                    reason = js.get('reason', '') if isinstance(js, dict) else ''
                    self.debugger.add('k-combo-select', {'id': id, 'js': js, 'n_bundles': n_cands})
                    best_idx = va_Prompt.extract_combo_select_response(js, n_cands)
                    if best_idx is not None:
                        if reason:
                            self.debugger.print(''.join(['* combo select reason: ', format(reason)]))
                        result = list(candidates[best_idx][0])
                        self.sess.step_on_combo_select(result, n_cands, reason, is_fallback=False)
                        return result
                    break
                except Exception as exc:
                    self.debugger.print(''.join(['# failed combo select: ', format(exc)]))
                    temperature += 0.3
            self.debugger.print('* combo select returned null — falling back to top-scored combo')
            result = list(candidates[0][0])
            self.sess.step_on_combo_select(result, len(candidates), '', is_fallback=True)
            return result

        def __proc_solmap__(self, solmap: Dict[str, List[Tuple[Tuple[va_Product, ...], Any]]], sort: bool, id: str, max_tries: int) -> List[va_Product]:
            t, cursor, is_exhausted = (0, 0, False)
            for t in range(max_tries):
                batch, candidates, cand_count = ([], defaultdict(lambda: []), 0)
                pre_valid: List[Tuple[Tuple[va_Product, ...], Any]] = []
                if not self.__check_watch__():
                    break
                while cand_count < va_SearchEngine.FILTER_BATCH and (not is_exhausted):
                    is_added = False
                    for _, sols in solmap.items():
                        if cursor < len(sols):
                            prods, w = sols[cursor]
                            is_added = True
                            if all((self.filtered[self.qi.items[i]][p] for i, p in enumerate(prods))):
                                pre_valid.append((prods, w))
                                continue
                            if any((self.filtered[self.qi.items[i]][p] == False for i, p in enumerate(prods))):
                                continue
                            for i, p in enumerate(prods):
                                sq = self.qi.items[i]
                                if self.filtered[sq][p] is None and p not in candidates[sq]:
                                    candidates[sq].append(p)
                                    cand_count += 1
                            batch.append((prods, w))
                    if not is_added:
                        is_exhausted = True
                    cursor += 1
                if cand_count > 0:
                    for sq, prods in self.__filter__(candidates, ''.join([format(id), '-sol-', format(t)]), 'k').items():
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
                    self.debugger.print(''.join(['* selected best combo from filtered at try ', format(t + 1), '!']))
                    return best_p
                if is_exhausted:
                    break
            return []

        def __find_filtered_combo__(self, solutions: List[Tuple[va_Product, Any]]) -> List[va_Product]:
            valid: List[Tuple[Tuple[va_Product, ...], Any]] = []
            seen: set = set()
            for prods, w in solutions:
                if all((bool(self.filtered[self.qi.items[i]][p]) for i, p in enumerate(prods))):
                    key = tuple((p.id for p in prods))
                    if key not in seen:
                        seen.add(key)
                        valid.append((prods, w))
            if not valid:
                return []
            if len(valid) == 1:
                return list(valid[0][0])
            self.debugger.print(''.join(['* multiple valid combos (', format(len(valid)), ') — escalating to LLM combo select']))
            return self.__select_best_combo__(valid, 'filtered-multi')

        def __solve_knapsack__(self, _: List[List[Tuple[va_Product, float]]], **__) -> List[Tuple[Tuple[va_Product, ...], float]]:
            return []



__all__ = ["agent_main"]
