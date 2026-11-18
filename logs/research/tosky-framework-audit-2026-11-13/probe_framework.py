"""Characterize the supplied framework offline; do not change its source."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT))
SOURCE = PROJECT/'agents/TOSKY.py'
DIGEST = '31b083e8ac95bc6863a4a01f9768a61879914fed4eb9116ea0f27b394d825a18'


def tool(name, fields=None):
    fields = fields or {}
    return {'type': 'function', 'function': {'name': name, 'description': name, 'parameters': {
        'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}}}


def problem():
    tools = [tool('find_catalog', {'query': {'type': 'string'}}), tool('read_item', {'id': {'type': 'string'}}),
             tool('read_offer', {'id': {'type': 'string'}}), tool('compare_items', {'ids': {'type': 'array', 'items': {'type': 'string'}}}),
             tool('ask_user', {'text': {'type': 'string'}}), tool('change_cart', {'id': {'type': 'string'}, 'quantity': {'type': 'integer'}}),
             tool('inspect_cart'), tool('order_now'), tool('stop_shopping')]
    return {'environment': {'binding': {'session_id': 'offline-session', 'route': 'opaque-route'},
            'policy_view': {'query': 'Buy one travel bag, with a matte surface, for no more than 50 credits.',
                            'tools': tools, 'max_steps': 20, 'finished': False}}}


def make(module):
    ep = module.Tosky(problem())
    ep.slots = [module.Slot(1, 'travel bag', 1, ['surface: matte'])]
    ep.roles = {'search': 'find_catalog', 'open': 'read_item', 'offer': 'read_offer', 'compare': 'compare_items',
                'message': 'ask_user', 'cart': 'change_cart', 'cart_view': 'inspect_cart',
                'checkout': 'order_now', 'decline': 'stop_shopping'}
    ep.budget, ep.budget_unit = Decimal(50), 'credits'
    ep.step = 1
    row = {'id': 'A', 'kind': 'travel bag', 'price': 10, 'stock': True, 'features': {'surface': 'matte'}}
    c = ep.get_candidate(1, 'A')
    c.row = deepcopy(row);c.detail = deepcopy(row);c.pages = {'read_item': deepcopy(row)}
    c.detail_step = c.offer_step = 1;c.price = Decimal(10);c.price_unit = 'credits';c.stock = True
    c.views_seen = {'read_item'}
    return ep, c


def judgment(*, requirement='surface: matte', path='features.surface', quote='matte'):
    return {'matches': [{'requirement': requirement, 'status': 'true', 'source': 'detail', 'path': path, 'quote': quote}],
            'stock': {'status': 'true', 'source': 'row', 'path': 'stock', 'quote': 'true'},
            'price': {'amount': 10, 'unit': 'credits', 'source': 'row', 'path': 'price', 'quote': '10'}}


def ready(ep, c):
    c.verdict = 'true';c.evaluated_revision = ep.revision
    ep.desired = {1: c.ident};ep.cart_lines = {c.ident: 1};ep.cart_dirty = False;ep.cart_step = 1


def record(rows, name, severity, lines, exercise):
    value = exercise()
    rows.append({'case': name, 'severity': severity, 'lines': lines, 'observed': value,
                 'scope': 'Deterministic controller counterexample using controlled interpreter replies; no paid inference or task grading.'})


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == DIGEST
    spec = importlib.util.spec_from_file_location('tosky_framework_audit', SOURCE)
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module
    rows, controls = [], []
    with patch('requests.sessions.Session.request', side_effect=AssertionError('Real network forbidden')) as network, \
         patch.dict('os.environ', {'SANDBOX_MODEL': ''}):
        spec.loader.exec_module(module)

        def price_and_type():
            ep, c = make(module);ep.slots[0].hard = []
            c.row['kind'] = c.detail['kind'] = 'camping stove'
            result = judgment();result['matches'] = []
            with patch.object(ep, 'infer', return_value=result):ep.evaluate(c)
            chosen = ep.choose_basket()
            assert c.verdict == 'true' and chosen == {1: 'A'}
            return {'requested_role': ep.slots[0].role, 'observed_kind': c.detail['kind'], 'verdict': c.verdict,
                    'selected': chosen, 'type_result_is_not_required_by_protocol': True}
        record(rows, 'requested_type_not_a_checked_requirement', 'P0', [797, 847, 966], price_and_type)

        def unknown():
            ep, c = make(module)
            with patch.object(ep, 'infer', return_value=None):ep.evaluate(c)
            with patch.object(ep, 'infer', return_value=judgment()) as again:ep.evaluate(c)
            assert c.verdict == 'unknown' and again.call_count == 0 and ep.next_unverified() is None
            return {'verdict': c.verdict, 'cached_as_current_revision': c.evaluated_revision == ep.revision,
                    'retry_interpreter_calls': again.call_count, 'next_unverified': None}
        record(rows, 'unavailable_evaluation_becomes_an_unrecoverable_current_unknown', 'P0', [794, 813, 883, 1239], unknown)

        def dotted_path():
            ep, c = make(module);c.detail['features'] = {'surface.finish': 'matte'}
            result = judgment(path='features.surface.finish')
            with patch.object(ep, 'infer', return_value=result):ep.evaluate(c)
            assert module.path_value(c.detail, 'features.surface.finish') is None and c.verdict == 'unknown'
            return {'observed_literal_key': 'surface.finish', 'actual_value': 'matte', 'path_result': None, 'verdict': c.verdict}
        record(rows, 'literal_dotted_attribute_keys_cannot_be_addressed_as_leaf_paths', 'P0', [90, 816], dotted_path)

        def comparison_identity():
            ep, c = make(module);ep.slots[0].hard = ['surface: gloss']
            other = deepcopy(c.detail);other['id'] = 'B';other['features']['surface'] = 'gloss'
            observation = {'items': [deepcopy(c.detail), other]}
            c.pages = {};c.detail = None
            ep.absorb_detail('compare_items', {'ids': ['A', 'B']}, observation, 'open item 1 target A')
            result = judgment(requirement='surface: gloss', path='items.1.features.surface', quote='gloss')
            with patch.object(ep, 'infer', return_value=result):ep.evaluate(c)
            assert c.pages['compare_items']['id'] == 'A' and c.verdict == 'true'
            return {'candidate': 'A', 'candidate_actual_surface': observation['items'][0]['features']['surface'],
                    'cited_entity': observation['items'][1]['id'], 'verdict': c.verdict,
                    'isolated_page_was_available_but_whole_response_used': True}
        record(rows, 'comparison_evidence_from_another_product_is_accepted', 'P0', [633, 635, 815, 837], comparison_identity)

        def price_non_numeric():
            ep, c = make(module);c.detail['price_note'] = 'Contact the seller for pricing.'
            result = judgment();result['price'] = {'amount': 1, 'unit': 'credits', 'source': 'detail',
                                                   'path': 'price_note', 'quote': 'Contact the seller'}
            with patch.object(ep, 'infer', return_value=result):ep.evaluate(c)
            assert c.price == 1 and ep.choose_basket() == {1: 'A'}
            return {'numeric_price_in_row': 10, 'nonnumeric_citation': c.detail['price_note'], 'accepted_price': str(c.price)}
        record(rows, 'unparseable_price_evidence_accepts_an_unsupported_model_amount', 'P0', [862, 863], price_non_numeric)

        def price_boolean():
            ep, c = make(module);result = judgment()
            result['price'] = {'amount': 1, 'unit': 'credits', 'source': 'row', 'path': 'stock', 'quote': 'true'}
            with patch.object(ep, 'infer', return_value=result):ep.evaluate(c)
            assert c.price == 1
            return {'cited_value': True, 'cited_value_type': 'bool', 'accepted_price': str(c.price)}
        record(rows, 'stock_boolean_can_be_used_as_price_evidence', 'P0', [815, 863], price_boolean)

        def stale_row():
            ep, c = make(module);ready(ep, c)
            result = {'shape': {'rows_path': 'results', 'id_key': 'id', 'price_key': 'price', 'stock_key': 'stock',
                                'count_path': 'total', 'next_path': 'next', 'price_unit': 'credits'}}
            observation = {'results': [{'id': 'A', 'price': 10, 'stock': True, 'features': {'surface': 'gloss'}}], 'total': 1, 'next': None}
            ep.absorb_search('find_catalog', {'query': 'travel bag'}, observation, 'search item 1', result)
            assert c.evaluated_revision == ep.revision and c.verdict == 'true' and ep.checkout_ready()
            return {'new_row_surface': c.row['features']['surface'], 'cached_detail_surface': c.detail['features']['surface'],
                    'cached_verdict': c.verdict, 'checkout_ready': ep.checkout_ready()}
        record(rows, 'changed_search_record_does_not_invalidate_cached_eligibility', 'P0', [607, 794, 1281], stale_row)

        def stale_offer():
            ep, c = make(module);ready(ep, c)
            old = deepcopy(c.detail);old['stock'] = True
            ep.step = 5
            current = deepcopy(old);current['stock'] = False
            ep.absorb_detail('read_offer', {'id': 'A'}, current, 'open item 1 target A')
            result = judgment(path='views.0.features.surface')
            result['stock'] = {'status': 'true', 'source': 'detail', 'path': 'views.0.stock', 'quote': 'true'}
            with patch.object(ep, 'infer', return_value=result):ep.evaluate(c)
            assert c.stock is True and c.offer_step == 5 and ep.checkout_ready()
            return {'latest_offer_stock': False, 'old_page_stock': True, 'accepted_stock': c.stock,
                    'old_evidence_stamped_with_step': c.offer_step, 'checkout_ready': ep.checkout_ready()}
        record(rows, 'old_stock_evidence_is_promoted_to_a_fresh_offer', 'P0', [634, 636, 856, 857], stale_offer)

        def cart_quantity():
            ep, c = make(module);ready(ep, c)
            observation = {'items': [{'id': 'A', 'quantity': 2, 'price': 1}]}
            ep.absorb_cart(observation, {'cart_lines': [{'id': 'A', 'quantity': 1}]})
            assert ep.cart_lines == {'A': 1} and ep.checkout_ready()
            return {'actual_quantity': 2, 'price_field': 1, 'accepted_quantity': ep.cart_lines['A'], 'checkout_ready': ep.checkout_ready()}
        record(rows, 'cart_quantity_is_grounded_by_an_unrelated_numeric_field', 'P0', [650, 653], cart_quantity)

        def cart_omission():
            ep, c = make(module);ready(ep, c)
            observation = {'items': [{'id': 'A', 'quantity': 1}, {'id': 'B', 'quantity': 1}]}
            ep.absorb_cart(observation, {'cart_lines': [{'id': 'A', 'quantity': 1}]})
            assert ep.cart_lines == {'A': 1} and ep.checkout_ready()
            return {'actual_line_ids': ['A', 'B'], 'accepted_line_ids': list(ep.cart_lines), 'checkout_ready': ep.checkout_ready()}
        record(rows, 'partial_cart_parser_output_is_accepted_as_an_exact_cart', 'P0', [645, 658, 1273], cart_omission)

        def reference_refresh():
            ep, c = make(module);ep.slots[0].hard = ['surface: same as the reference item']
            with patch.object(ep, 'infer', return_value={}):ep.evaluate(c)
            ep.references.append({'id': 'R', 'features': {'surface': 'matte'}})
            with patch.object(ep, 'infer', return_value=judgment()) as retry:ep.evaluate(c)
            assert c.verdict == 'unknown' and retry.call_count == 0
            return {'reference_pages_now_available': len(ep.references), 'verdict': c.verdict,
                    'reevaluation_calls_after_new_reference': retry.call_count}
        record(rows, 'new_reference_pages_do_not_invalidate_an_unknown_candidate', 'P0', [565, 794, 833, 883], reference_refresh)

        def false_reference():
            ep, c = make(module);ep.slots[0].hard = ['weight: within the allowed bound']
            c.detail['features']['weight'] = 8
            result = judgment(requirement=ep.slots[0].hard[0], path='features.weight', quote='8')
            with patch.object(ep, 'infer', return_value=result):ep.evaluate(c)
            assert c.verdict == 'unknown'
            return {'literal_word_within_requires_reference': True, 'reference_required_by_task': False, 'verdict': c.verdict}
        record(rows, 'ordinary_within_bounds_requirement_is_misclassified_as_a_reference', 'P1', [831, 834], false_reference)

        def drop_ungrounded():
            ep, c = make(module)
            ep.apply_contract_delta({'drop': [{'item': 1, 'property': 'surface', 'quote': 'absent quotation'}]}, 'No change is requested.')
            assert ep.slots[0].hard == []
            return {'source': 'No change is requested.', 'remaining_requirements': ep.slots[0].hard}
        record(rows, 'a_dropped_requirement_has_no_source_quote_validation', 'P0', [759, 768], drop_ungrounded)

        def replacement_erases_new():
            ep, c = make(module)
            source = 'Use gloss instead of matte for the surface.'
            delta = {'add': [{'item': 1, 'requirement': 'surface: gloss', 'quote': 'gloss'}],
                     'drop': [{'item': 1, 'property': 'surface', 'quote': source}]}
            ep.apply_contract_delta(delta, source)
            assert ep.slots[0].hard == []
            return {'source': source, 'expected_active_need': 'surface: gloss', 'actual_active_needs': ep.slots[0].hard,
                    'new_need_was_added_then_removed_by_the_same_drop': True}
        record(rows, 'same_reply_replacement_erases_both_the_old_and_new_requirement', 'P0', [746, 757, 759, 768], replacement_erases_new)

        def units_and_quantity_delta():
            ep, c = make(module)
            ep.apply_contract_delta({'budget': 5, 'budget_unit': 'hundredths', 'budget_quote': '5 hundredths',
                                     'quantity': [{'item': 1, 'quantity': 3}], 'free': [{'item': 1, 'property': 'surface'}]},
                                    'The limit is 5 hundredths; take 3; I no longer mind the surface.')
            assert ep.budget == 5 and ep.budget_unit == 'credits' and ep.slots[0].quantity == 1 and ep.slots[0].hard
            return {'budget_amount': str(ep.budget), 'budget_unit_retained': ep.budget_unit,
                    'quantity_retained': ep.slots[0].quantity, 'free_field_ignored': ep.slots[0].hard}
        record(rows, 'budget_units_quantity_and_free_state_are_not_applied', 'P1', [737, 745, 770], units_and_quantity_delta)

        def missing_quote_change():
            ep, c = make(module)
            ep.absorb_changes([{'kind': 'budget', 'amount': 500, 'text': ''}], {'notice': 'The catalog was updated.'})
            assert ep.budget == 500
            return {'observation': {'notice': 'The catalog was updated.'}, 'accepted_budget': str(ep.budget)}
        record(rows, 'empty_market_change_quote_can_authorize_a_budget_revision', 'P1', [711, 718], missing_quote_change)

        def question_limit():
            ep, c = make(module);ep.message_limit = 8;ep.messages_used = 2
            ep.questions = [{'item': 1, 'property': 'carrying comfort', 'mandatory': False}]
            assert ep.question_to_ask() is None
            return {'message_limit': ep.message_limit, 'messages_used': ep.messages_used,
                    'unasked_optional_questions': 1, 'selected_question': None}
        record(rows, 'two_optional_messages_stop_discovery_despite_unused_budget', 'P1', [1068], question_limit)

        def duties_reservation():
            ep, c = make(module);ep.message_limit = 2
            ep.obligations = [module.Obligation(0, 'Send the requested summary', 'before_order', 'message'),
                              module.Obligation(1, 'Send the requested recommendation', 'before_order', 'message')]
            ep.questions = [{'item': 1, 'property': 'carrying comfort', 'mandatory': False}]
            assert ep.question_to_ask() is not None
            return {'message_limit': 2, 'mandatory_message_duties': 2, 'optional_question_admitted': True}
        record(rows, 'nonapproval_message_duties_are_not_reserved', 'P1', [1063, 1066], duties_reservation)

        def confused_reply():
            ep, c = make(module)
            ep.pending_question = {'item': 1, 'property': 'surface', 'mandatory': True, 'obligation': 0}
            duty = module.Obligation(0, 'Ask about surface', 'before_selection', 'question');ep.obligations = [duty]
            reply = 'Please rephrase that question; I do not understand it.'
            with patch.object(ep, 'infer', return_value={}):ep.absorb_reply(reply, {'reply': reply})
            assert '1:surface' in ep.asked and duty.answered
            return {'reply': reply, 'property_marked_asked': True, 'mandatory_question_marked_answered': True}
        record(rows, 'a_clarification_request_is_counted_as_an_answered_question', 'P1', [674, 678], confused_reply)

        def premature_order():
            ep, c = make(module);ready(ep, c);ep.objective = 'cheapest'
            other = ep.get_candidate(1, 'B');other.row = {'id': 'B'};other.price = Decimal(5);other.stock = True
            ep.scopes['remaining'] = module.Scope('find_catalog', {}, 1, complete=False)
            assert ep.unresolved_better() and ep.checkout_ready()
            return {'unresolved_better': True, 'cheaper_unassessed_price': str(other.price), 'checkout_ready': True}
        record(rows, 'checkout_gate_does_not_enforce_the_optimality_frontier', 'P1', [1263, 1277, 1288, 1411], premature_order)

        def mixed_measure_units():
            ep, c = make(module);ep.slots[0].hard = [];ready(ep, c)
            ep.objective = 'most';ep.objective_item = 1;c.measure = Decimal(2);c.measure_unit = 'kg'
            other = ep.get_candidate(1, 'B');other.row = {'id': 'B'};other.detail = {'id': 'B'}
            other.price = Decimal(10);other.price_unit = 'credits';other.stock = True
            other.verdict = 'true';other.evaluated_revision = ep.revision
            other.measure = Decimal(900);other.measure_unit = 'g'
            assert ep.choose_basket() == {1: 'B'}
            return {'A': {'measure': '2', 'unit': c.measure_unit}, 'B': {'measure': '900', 'unit': other.measure_unit},
                    'selected_by_raw_number': 'B', 'no_measure_unit_normalization_is_called': True}
        record(rows, 'extreme_objectives_compare_raw_numbers_across_measure_units', 'P1', [890, 984], mixed_measure_units)

        def too_many_rows():
            ep, c = make(module);ep.candidates = {}
            observation = {'results': [{'id': f'item-{i}'} for i in range(81)], 'total': 81, 'next': None}
            ep.absorb_search('find_catalog', {}, observation, 'search item 1', {'shape': {
                'rows_path': 'results', 'id_key': 'id', 'count_path': 'total', 'next_path': 'next'}})
            scope = next(iter(ep.scopes.values()))
            assert len(scope.seen) == 80 and scope.complete and len(ep.candidates) == 80
            return {'rows_returned': 81, 'candidates_retained': 80, 'scope_marked_complete': True}
        record(rows, 'row_cap_silently_drops_evidence_while_marking_scope_complete', 'P1', [596, 622], too_many_rows)

        def failed_ack():
            ep, c = make(module)
            assert not ep.observation_error({'accepted': False}) and not ep.observation_error({'success': False})
            return {'accepted_false_treated_as_error': False, 'success_false_treated_as_error': False}
        record(rows, 'negative_acknowledgements_are_not_classified_as_action_failure', 'P1', [776, 1174], failed_ack)

        def required_selection():
            ep, c = make(module);ready(ep, c);ep.desired = {};ep.cart_lines = {}
            duty = module.Obligation(0, 'Obtain permission before selecting', 'before_selection', 'approval')
            ep.obligations = [duty];calls = []
            def align():
                calls.append({'duty_answered': duty.answered});ep.step = ep.max_steps;return False
            with patch.object(ep, 'bootstrap'), patch.object(ep, 'discover_questions'), \
                 patch.object(ep, 'do_process', return_value=False), patch.object(ep, 'align_cart', side_effect=align):
                ep.run()
            assert calls and calls[0]['duty_answered'] is False
            return {'cart_alignment_reached_before_required_permission': True}
        record(rows, 'failed_before_selection_duty_does_not_block_cart_alignment', 'P0', [1382, 1391, 1435], required_selection)

        def mandatory_limit():
            ep, c = make(module);ep.message_limit = 0
            ep.questions = [{'item': 1, 'property': 'surface', 'mandatory': True, 'obligation': 0}]
            ep.obligations = [module.Obligation(0, 'Ask about surface', 'before_selection', 'question')]
            attempts = []
            def perform(intent, names, **kwargs):
                attempts.append(intent);ep.finished = True;return True
            with patch.object(ep, 'bootstrap'), patch.object(ep, 'discover_questions'), patch.object(ep, 'evaluate'), \
                 patch.object(ep, 'perform', side_effect=perform):ep.run()
            assert attempts == ['question']
            return {'message_limit': 0, 'mandatory_question_attempted': True}
        record(rows, 'mandatory_question_branch_ignores_the_message_ceiling', 'P1', [1386, 1387], mandatory_limit)

        def read_duty_key():
            ep, c = make(module);ready(ep, c)
            duty = module.Obligation(0, 'Read the selected page before ordering', 'before_order', 'read')
            ep.obligations = [duty]
            def infer(purpose, *_, **__):
                if purpose == 'observation':return {}
                if purpose == 'process proof':return {'satisfied': True, 'quote': 'A'}
                if purpose == 'candidate evidence':return judgment()
                raise AssertionError(purpose)
            world = SimpleNamespace(post_verbose=lambda path, body: SimpleNamespace(
                data={'step': body['step'], 'finished': False, 'observation': deepcopy(c.row)}, error=None))
            with patch.object(ep, 'plan_action', return_value=('read_item', {'id': 'A'})), \
                 patch.object(ep, 'infer', side_effect=infer), patch.object(module, '_environment', world):
                assert ep.perform('procedure', ['read_item'], obligation=duty)
                before = ep.obligation_satisfied(duty);ep.evaluate(c);after = ep.obligation_satisfied(duty)
            assert before and not after
            return {'performed_read_was_satisfied': before, 'satisfied_after_evaluating_same_page': after,
                    'price_and_stock_materially_changed': False}
        record(rows, 'successful_required_read_invalidates_its_own_basket_proof', 'P0', [857, 919, 1048, 1186], read_duty_key)

        def message_fulfills_read():
            ep, c = make(module);ready(ep, c)
            duty = module.Obligation(0, 'Read the selected page', 'before_order', 'read');ep.obligations = [duty]
            def infer(purpose, *_, **__):
                return {'reply': 'Hello.', 'changes': []} if purpose == 'observation' else {}
            world = SimpleNamespace(post_verbose=lambda path, body: SimpleNamespace(
                data={'step': body['step'], 'finished': False, 'observation': 'Hello.'}, error=None))
            with patch.object(ep, 'plan_action', return_value=('ask_user', {'text': 'Hello.'})), \
                 patch.object(ep, 'infer', side_effect=infer), patch.object(module, '_environment', world):
                assert ep.do_process(duty)
            assert ep.obligation_satisfied(duty)
            return {'required_action_kind': 'read', 'executed_tool': 'ask_user', 'read_obligation_satisfied': True}
        record(rows, 'a_message_reply_can_satisfy_a_read_obligation', 'P0', [683, 686, 1173, 1307], message_fulfills_read)

        def no_money_stop():
            ep, c = make(module);posted = []
            def post(path, body):
                posted.append(body)
                return SimpleNamespace(data={'usage': {'cost': .06}, 'choices': [{'message': {'content': '{}'}}]}, error=None)
            with patch.object(module, '_inference', SimpleNamespace(post_verbose=post)):
                for _ in range(3):assert ep.infer('cost counterexample', 'Synthetic accounting data') == {}
            assert len(posted) == 3 and ep.model_cost > .17
            return {'physical_requests_after_already_exceeding_5_divided_by_90': len(posted),
                    'controlled_reported_cost_usd': ep.model_cost, 'per_episode_dollar_limit': None,
                    'limitation': 'Controlled billing replies, not an estimate of real model prices or expected spend.'}
        record(rows, 'reported_spend_is_never_used_as_an_admission_limit', 'P0', [320, 345, 346], no_money_stop)

        def credit_stop():
            ep, c = make(module);posted = []
            def post(path, body):
                posted.append(body);return SimpleNamespace(data=None, error={'status': 402})
            with patch.object(module, '_inference', SimpleNamespace(post_verbose=post)):
                assert ep.infer('credit', 'x') is None;assert ep.infer('credit again', 'x') is None
            assert len(posted) == 4
            return {'requests_for_two_logical_calls': len(posted), 'requests_after_first_402': 3}
        record(rows, 'credit_402_does_not_stop_backup_or_later_inference', 'P0', [337, 342], credit_stop)

        def terminal_inference():
            ep, c = make(module);ready(ep, c);calls = []
            world = SimpleNamespace(post_verbose=lambda path, body: SimpleNamespace(
                data={'step': body['step'], 'finished': True, 'observation': {'accepted': True}}, error=None))
            with patch.object(ep, 'plan_action', return_value=('order_now', {})), \
                 patch.object(ep, 'infer', side_effect=lambda purpose, *a, **k: calls.append(purpose) or {}), \
                 patch.object(module, '_environment', world):assert ep.terminal('buy')
            assert ep.finished and calls == ['observation']
            return {'world_finished': True, 'interpreter_calls_after_terminal_receipt': calls}
        record(rows, 'finished_terminal_receipt_still_triggers_observation_inference', 'P1', [545, 547, 559, 1172], terminal_inference)

        def relation_duplicate():
            ep, c = make(module);ep.relations = ['relation one', 'relation two']
            result = {'fits': True, 'proof': [{'relation': 'relation one', 'quote': 'matte'},
                                             {'relation': 'relation one', 'quote': 'matte'}]}
            with patch.object(ep, 'infer', return_value=result):accepted = ep.relation_ok({1: 'A'})
            assert accepted
            return {'required_relations': ep.relations, 'cited_relations': ['relation one', 'relation one'], 'accepted': accepted}
        record(rows, 'duplicate_relation_proofs_can_cover_an_omitted_relation', 'P1', [956, 958], relation_duplicate)

        def malformed_contract():
            ep, c = make(module);ep.slots = []
            with patch.object(ep, 'infer', return_value={'items': [{'role': 'travel bag', 'hard': 'matte'}]}):ep.bootstrap()
            assert ep.slots[0].hard == list('matte')
            return {'returned_hard_field': 'matte', 'accepted_requirements': ep.slots[0].hard}
        record(rows, 'a_string_in_contract_hard_becomes_character_requirements', 'P1', [423, 425], malformed_contract)

        def schema():
            spec = {'type': 'object', 'properties': {'code': {'type': 'string', 'pattern': '^[A-Z]{2}$', 'minLength': 2}}, 'required': ['code']}
            errors = module.valid_schema({'code': 'x'}, spec)
            assert errors == []
            return {'invalid_argument': {'code': 'x'}, 'schema_pattern': '^[A-Z]{2}$', 'local_errors': errors}
        record(rows, 'string_pattern_and_length_constraints_are_not_validated', 'P2', [151, 191], schema)

        def helper_numbers():
            assert module.number(float('nan')).is_nan()
            assert module.number('1,25') == Decimal(125)
            assert module.stock_value('not in stock') is True
            return {'nonfinite_number_retained': True, 'comma_decimal_1_25_parsed_as': '125', 'not_in_stock_parsed_as': True}
        record(rows, 'numeric_and_stock_helpers_accept_ambiguous_or_invalid_values', 'P1', [66, 69, 146], helper_numbers)

        ep, c = make(module)
        with patch.object(ep, 'infer', return_value=judgment()):ep.evaluate(c)
        assert c.verdict == 'true' and ep.choose_basket() == {1: 'A'}
        controls.append('valid_same_product_evidence_selects_the_candidate')

        ep, c = make(module);bad = judgment();bad['matches'][0]['quote'] = 'not observed here'
        with patch.object(ep, 'infer', return_value=bad):ep.evaluate(c)
        assert c.verdict == 'unknown'
        controls.append('an_absent_quote_is_not_positive_evidence')

        ep, c = make(module);ep.budget = Decimal(5)
        with patch.object(ep, 'infer', return_value=judgment()):ep.evaluate(c)
        assert ep.choose_basket() == {}
        controls.append('a_known_over_budget_candidate_is_not_selected')

        ep, c = make(module);c.row['stock'] = False;answer = judgment();answer['stock'].update(status='false', quote='false')
        with patch.object(ep, 'infer', return_value=answer):ep.evaluate(c)
        assert c.verdict == 'false' and ep.choose_basket() == {}
        controls.append('grounded_unavailable_stock_excludes_the_candidate')

        ep, c = make(module);ready(ep, c)
        ep.obligations = [module.Obligation(0, 'Approve the cart', 'before_order', 'approval')]
        assert not ep.checkout_ready()
        controls.append('an_unanswered_before_order_obligation_blocks_checkout')

        ep, c = make(module)
        ep.apply_contract_delta({'add': [{'item': 1, 'requirement': 'surface: gloss', 'quote': 'absent quote'}]}, 'No change.')
        assert ep.slots[0].hard == ['surface: matte']
        controls.append('an_explicitly_ungrounded_addition_is_rejected')

        ep, c = make(module);attempts = []
        def replay(path, body):
            attempts.append(deepcopy(body))
            if len(attempts) == 1:raise TimeoutError('controlled uncertain transport')
            return SimpleNamespace(data={'step': body['step'], 'observation': c.row, 'finished': False}, error=None)
        with patch.object(module, '_environment', SimpleNamespace(post_verbose=replay)):
            assert ep.execute('read_item', {'id': 'A'}) is not None
        assert len(attempts) == 2 and attempts[0] == attempts[1]
        assert attempts[0]['route'] == 'opaque-route' and ep.step == 2 and len(ep.events) == 1
        controls.append('uncertain_environment_retry_preserves_binding_step_and_action')

        ep, c = make(module);attempts = []
        with patch.object(module, '_environment', SimpleNamespace(post_verbose=lambda *a: attempts.append(a))):
            assert ep.execute('change_cart', {'id': 'A', 'quantity': True}) is None
        assert attempts == []
        controls.append('boolean_quantity_is_rejected_before_environment_dispatch')

        ep, c = make(module);c.price_unit = 'hundredths';c.price = Decimal(100)
        with patch.object(ep, 'infer', return_value={'compatible': True, 'factor': .01}) as infer:
            assert ep.basket_total({1: 'A'}) == Decimal(1)
            assert ep.basket_total({1: 'A'}) == Decimal(1) and infer.call_count == 1
        controls.append('runtime_money_unit_conversion_is_cached_without_a_fixed_registry')

        network.assert_not_called()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == DIGEST
    output = {'source': str(SOURCE.relative_to(PROJECT)), 'sha256': DIGEST, 'source_changed': False,
              'network_calls': 0, 'paid_calls': 0, 'counterexamples_confirmed': len(rows),
              'positive_controls_passed': len(controls), 'positive_controls': controls, 'cases': rows}
    (ROOT/'counterexamples.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':main()
