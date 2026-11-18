"""Offline checks for the new candidate; real network requests are forbidden."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_chidori_v3 import load, problem, response, Client, OfflineCase

ROOT = Path(__file__).resolve().parents[2]
agent = load(ROOT/'agents/Tosky_v2.py', 'tosky_v2_candidate')


def episode(data=None):
    ep = agent.Episode(data or problem())
    ep.room = lambda: 100
    ep.clock = lambda: 0
    ep.brief_read = True
    ep.procedures_valid = True
    ep.coverage_done = True
    ep.items = [agent.Item('equipment', 'equipment', 1, [])]
    ep.roles = {'open': 'read', 'buy': 'buy', 'refuse': 'decline', 'ask': 'message'}
    return ep


def tool_reply(name='read', args=None, cost=.000001):
    return SimpleNamespace(data={'usage': {'cost': cost}, 'choices': [{'message': {
        'tool_calls': [{'id': 'offline-call', 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(args or {'id': 'observed'})}}]}, 'finish_reason': 'tool_calls'}]}, error=None)


class BudgetAndContinuationTests(OfflineCase):
    def test_ordinary_success_does_not_enter_compact_path(self):
        ep = episode(); client = Client([tool_reply()])
        messages = [{'role': 'user', 'content': 'Continue normally.'}]
        with patch.object(agent, 'model_client', return_value=client):
            self.assertIsNotNone(ep.think(messages, [agent.LIGHT], 700, ep.tools))
        self.assertEqual(ep.compact_count, 0)
        self.assertEqual(client.posts[0][1]['messages'], messages)
        self.assertNotIn('reasoning', client.posts[0][1])

    def test_unaffordable_history_gets_a_real_accounted_compact_continuation(self):
        ep = episode(); ep.request = 'Find suitable equipment.'
        ep.rows = {'observed': {'id': 'observed', 'attributes': {'finish': 'satin'}}}; ep.order = ['observed']
        client = Client([tool_reply()])
        with patch.object(agent, 'model_client', return_value=client):
            answer = ep.think([{'role': 'user', 'content': 'x'*50000}], [agent.HEAVY], 3000, ep.tools)
        self.assertIsNotNone(answer)
        self.assertEqual(ep.compact_count, 1)
        self.assertEqual(ep.calls, 1)
        self.assertEqual(len(client.posts), 1)
        body = client.posts[0][1]
        self.assertEqual(body['reasoning'], {'enabled': False})
        self.assertIn('observed', json.dumps(body['messages'][1]['content']))
        self.assertFalse(ep.request_denied)
        self.assertEqual(ep.inference_budget.pending, 0)

    def test_remote_credit_denial_cannot_trigger_a_compact_retry(self):
        ep = episode();client = Client([SimpleNamespace(data=None, error={'status': 402})])
        with patch.object(agent, 'model_client', return_value=client):
            self.assertIsNone(ep.think([{'role': 'user', 'content': 'Continue.'}], [agent.LIGHT], 700, ep.tools))
        self.assertTrue(ep.no_credit)
        self.assertEqual(ep.compact_count, 0)
        self.assertEqual(len(client.posts), 1)

    def test_new_roles_bound_physical_attempts_and_keep_unknown_charges(self):
        ep = episode();bad = SimpleNamespace(data=None, error={'status': 500})
        client = Client([bad, bad])
        with patch.object(agent, 'model_client', return_value=client), patch.object(agent.time, 'sleep'):
            self.assertIsNone(ep.with_inference_role('proof', ep.json_ask, 'Inspect public evidence.', [agent.LIGHT, agent.SPARE], 1600))
        self.assertEqual(ep.calls, 2)
        self.assertEqual(len(client.posts), 2)
        self.assertGreater(ep.inference_budget.uncertain, 0)
        self.assertEqual(ep.inference_budget.pending, 0)

    def test_exhausted_local_ledger_does_not_send_any_request(self):
        ep = episode();ep.inference_budget.actual = ep.inference_budget.limit;client = Client([])
        with patch.object(agent, 'model_client', return_value=client):
            self.assertIsNone(ep.think([{'role': 'user', 'content': 'Continue.'}], [agent.LIGHT], 700, ep.tools))
        self.assertEqual(client.posts, [])


class IdentityAndProtocolTests(OfflineCase):
    def test_nested_unobserved_identity_is_held_before_using_a_step(self):
        ep = episode();ep.roles['cart'] = 'basket'
        schema = {'type': 'object', 'properties': {'lines': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'reference': {'type': 'string'}, 'amount': {'type': 'integer'}}}}}}
        ep.schemas['basket'] = schema
        ep.rows = {'observed': {'id': 'observed'}}
        with patch.object(agent.ENV_CLIENT, 'post_verbose') as world:
            self.assertIsNone(ep.env('basket', {'lines': [{'reference': 'unobserved', 'amount': 1}]}))
        world.assert_not_called();self.assertEqual(ep.step, 0);self.assertEqual(ep.cart_lines, {})
        self.assertIn('unobserved', ep.evidence_fault)

    def test_free_text_that_is_not_an_identity_is_not_rejected(self):
        ep = episode()
        schema = {'type': 'object', 'properties': {'note': {'type': 'string', 'description': 'A basket note.'}}}
        self.assertEqual(ep.cart_identity_fault({'note': 'Handle carefully.'}, schema), '')

    def test_known_identity_and_enum_action_are_not_confused(self):
        ep = episode();ep.rows = {'observed': {'id': 'observed'}}
        schema = {'properties': {'reference': {'type': 'string'}, 'operation': {'type': 'string', 'enum': ['plus', 'minus']}}}
        self.assertEqual(ep.cart_identity_fault({'reference': 'observed', 'operation': 'plus'}, schema), '')

    def test_full_opaque_binding_and_consecutive_steps_reach_the_transport(self):
        data = problem();data['environment']['binding'].update(route='opaque-route', nonce='opaque-nonce')
        ep = episode(data);events = []
        def world(path, body):
            events.append(body)
            return SimpleNamespace(data={'step': body['step'], 'finished': False, 'observation': {'value': 'observed'}}, error=None)
        with patch.object(agent.ENV_CLIENT, 'post_verbose', side_effect=world), patch.object(agent.time, 'sleep'):
            ep.env('read', {'id': 'observed'});ep.env('read', {'id': 'observed'})
        self.assertEqual([e['step'] for e in events], [1, 2])
        self.assertTrue(all(e['route'] == 'opaque-route' and e['nonce'] == 'opaque-nonce' for e in events))


class SourceTests(unittest.TestCase):
    def test_original_full_v4_is_byte_identical(self):
        self.assertEqual(hashlib.sha256((ROOT/'agents/ICDI_chidori_v4.py').read_bytes()).hexdigest(),
                         'bbf30300c5039956ddd18faa5c22fd27b126458339f8ae67860c87425777e898')

    def test_new_controller_is_on_the_live_entrypoint_path(self):
        self.assertTrue(issubclass(agent.Episode, agent.ToskyControls))
        tree = ast.parse((ROOT/'agents/Tosky_v2.py').read_text())
        entries = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'agent_main']
        self.assertEqual(len(entries), 1);self.assertIsInstance(entries[0], ast.FunctionDef)
        self.assertEqual(agent.AGENT_USD_LIMIT, '0.035')


class IntakeTests(OfflineCase):
    def audit(self, constraints, **extra):
        return {'complete': True, 'procedures_complete': True, 'constraints': constraints, 'add': [], 'process': [], **extra}

    def clause(self, previous, need=None, source='The surface must be satin.', **extra):
        return {'item': 1, 'previous': previous, 'need': need if need is not None else previous,
                'source': source, 'relation': 'literal', **extra}

    def test_atomic_audit_can_remove_invented_context_and_preserve_explicit_requirement(self):
        ep = episode();ep.request = 'The surface must be satin. The old one took its final bow.'
        ep.items[0].musts = ['surface: satin', 'occasion: performance']
        answer = self.audit([self.clause('surface: satin'), self.clause('occasion: performance', source='The old one took its final bow.', kind='context')])
        answer['constraints'][1]['need'] = None
        self.assertTrue(ep.apply_source_audit(answer))
        self.assertEqual(ep.items[0].musts, ['surface: satin'])
        self.assertEqual(ep.version, 1)

    def test_one_bad_quote_rejects_entire_audit_without_partial_mutation(self):
        ep = episode();ep.request = 'The surface must be satin.';ep.items[0].musts = ['surface: satin']
        answer = self.audit([self.clause('surface: satin', need='surface: polished')],
                            add=[{'item': 1, 'previous': None, 'need': 'shape: square', 'source': 'invented text', 'relation': 'literal'}])
        self.assertFalse(ep.apply_source_audit(answer));self.assertEqual(ep.items[0].musts, ['surface: satin'])
        self.assertEqual(ep.version, 0)

    def test_audit_cannot_omit_a_live_constraint(self):
        ep = episode();ep.request = 'The surface must be satin.';ep.items[0].musts = ['surface: satin', 'edge: not sharp']
        self.assertFalse(ep.apply_source_audit(self.audit([self.clause('surface: satin')])))

    def test_explicit_unavailable_property_value_is_preserved(self):
        ep = episode();ep.request = 'The surface must be satin.';ep.items[0].musts = ['surface: satin']
        ep.rows = {'one': {'attributes': {'surface': 'matte'}}}
        self.assertTrue(ep.apply_source_audit(self.audit([self.clause('surface: satin')])) )
        self.assertEqual(ep.items[0].musts, ['surface: satin'])

    def test_capability_relation_and_alternatives_survive_intake(self):
        ep = episode();ep.request = 'The working range must cover +/- 8. The finish is satin or matte.'
        ep.items[0].musts = ['working range: up to +/- 8', 'finish: satin or matte']
        a = self.audit([self.clause('working range: up to +/- 8', source='The working range must cover +/- 8.', relation='coverage'),
                        self.clause('finish: satin or matte', source='The finish is satin or matte.')])
        self.assertTrue(ep.apply_source_audit(a));self.assertEqual(ep.need_evidence[(1, 'working range: up to +/- 8')]['relation'], 'coverage')
        self.assertIn('finish: satin or matte', ep.items[0].musts)

    def test_omitted_procedure_is_installed_from_actual_source(self):
        ep = episode();ep.request = 'Ask for approval before ordering.'
        duty = {'step': 'Obtain current basket approval', 'when': 'order', 'messages': 1, 'actions': 1,
                'source': ep.request, 'cart_dependent': True, 'group': '', 'combine_source': ''}
        self.assertTrue(ep.apply_source_audit(self.audit([], process=[duty])))
        self.assertEqual(ep.duties[0]['stage'], 'order');self.assertEqual(ep.free_msgs(), 0)
        self.assertTrue(ep.gate_due('order'))

    def test_malformed_previous_and_invented_group_permission_are_rejected(self):
        ep = episode();ep.request = 'Ask for approval before ordering.'
        self.assertFalse(ep.apply_source_audit(self.audit([{'item': 1, 'previous': [], 'need': 'a: b', 'source': ep.request, 'relation': 'literal'}])))
        duty = {'step': 'Approve', 'when': 'order', 'messages': 1, 'actions': 1,
                'source': ep.request, 'group': 'shared', 'combine_source': 'not in source'}
        self.assertFalse(ep.apply_source_audit(self.audit([], process=[duty])))

    def test_withdrawn_requirement_is_not_restored_by_an_addition(self):
        ep = episode();ep.request = 'The surface must be satin.';ep.items[0].dropped = ['surface: satin']
        answer = self.audit([], add=[{'item': 1, 'previous': None, 'need': 'surface: satin', 'source': ep.request, 'relation': 'literal'}])
        self.assertFalse(ep.apply_source_audit(answer))

    def test_reply_value_is_separated_from_framing(self):
        ep = episode();sent = [(1, 'Which surface?', 'Keep the surface fixed, with satin and nothing different.')]
        a = {'item': 1, 'property': 'surface', 'value': 'satin', 'operation': 'eq', 'source': sent[0][2]}
        self.assertEqual(ep.normalize_answer_need(a, sent)['need'], 'surface: satin')

    def test_reply_alternatives_exclusion_and_bound_keep_polarity(self):
        ep = episode();sent = [(1, 'Which surface?', 'Satin or matte. Exclude gloss. At most 8 mm.')]
        for value, op, expected in [(['Satin', 'matte'], 'one_of', 'surface: Satin or matte'),
                                    ('gloss', 'exclude', 'surface: not gloss'), ('At most 8 mm', 'constraint', 'surface: At most 8 mm')]:
            a = {'item': 1, 'property': 'surface', 'value': value, 'operation': op, 'source': sent[0][2]}
            self.assertEqual(ep.normalize_answer_need(a, sent)['need'], expected)

    def test_reply_cannot_invent_value_or_clamp_an_invalid_item(self):
        ep = episode();sent = [(1, 'Which surface?', 'Satin please.')]
        a = {'item': 1, 'property': 'surface', 'value': 'gloss', 'operation': 'eq', 'source': sent[0][2]}
        self.assertIsNone(ep.normalize_answer_need(a, sent));a.update(item=0, value='Satin')
        self.assertIsNone(ep.normalize_answer_need(a, sent))

    def test_bad_reply_stays_unread_and_can_be_repaired_once(self):
        ep = episode();sent = [(1, 'Which surface?', 'Satin please.')]
        valid = {'needs': [{'item': 1, 'property': 'surface', 'value': 'Satin', 'operation': 'eq', 'source': 'Satin please.'}],
                 'drop': [], 'free': [], 'crowded': [], 'budget': None}
        with patch.object(ep, 'evidence_decision', side_effect=[None, valid]) as decision:
            self.assertEqual(ep.read_answers(sent), {});self.assertEqual(ep.unread_replies, sent)
            answer = ep.read_answers(sent);self.assertEqual(answer['needs'][0]['need'], 'surface: Satin')
            self.assertEqual(ep.unread_replies, [])
            self.assertEqual(ep.read_answers(sent), {})
        self.assertEqual(decision.call_count, 2)


class InterviewTests(OfflineCase):
    def test_neighboring_and_component_properties_are_not_settled_by_shared_words(self):
        ep = episode();ep.items[0].musts = ['package type: individual', 'frame color: silver']
        self.assertTrue(ep.topic_settled(1, 'package_type'))
        self.assertFalse(ep.topic_settled(1, 'package size'))
        self.assertFalse(ep.topic_settled(1, 'lens color'))

    def test_topic_fallback_uses_current_observations_without_property_registry(self):
        ep = episode()
        table = [('equipment', 'zeta surface', 5, ['rough', 'satin']), ('equipment', 'alpha amount', 5, ['2 kg', '3 kg'])]
        with patch.object(ep, 'key_table', return_value=table), patch.object(ep, 'owner', return_value=1), \
             patch.object(ep, 'item_records', return_value=5), patch.object(ep, 'adopt'), patch.object(ep, 'evidence_decision', return_value=None):
            ep.plan(3)
        self.assertEqual(ep.lanes[1], ['zeta surface', 'alpha amount'])

    def test_model_cannot_add_invented_topic_or_replace_scope(self):
        ep = episode();ep.items[0].musts = ['frame color: silver']
        table = [('equipment', 'lens color', 4, ['clear', 'tinted'])]
        answer = {'ask': [{'item': 1, 'label': 'frame color'}, {'item': 1, 'label': 'invented'}, {'item': 1, 'label': 'lens_color'}]}
        with patch.object(ep, 'key_table', return_value=table), patch.object(ep, 'owner', return_value=1), \
             patch.object(ep, 'item_records', return_value=4), patch.object(ep, 'adopt'), patch.object(ep, 'evidence_decision', return_value=answer):
            ep.plan(3)
        self.assertEqual(ep.lanes[1], ['lens color'])

    def test_short_questions_and_crowding_feedback_control_width(self):
        ep = episode();ep.msg_limit = 3
        self.assertEqual(ep.list_now(), 1)
        ep.msg_used = 1;self.assertEqual(ep.list_now(), 1)
        ep.msg_used = 2;self.assertEqual(ep.list_now(), 2)
        ep.question_batches = [{'reading': {'crowded': [1]}}]
        self.assertEqual(ep.list_now(), 1)

    def test_procedure_messages_are_reserved_before_selecting_interview_width(self):
        ep = episode();ep.msg_limit = 3
        ep.duties = [{'id': 'd1', 'stage': 'order', 'messages': 1, 'actions': 1, 'completed': False,
                      'cart_dependent': False, 'group': '', 'signature': None}]
        ep.msg_used = 1
        self.assertEqual(ep.free_msgs(), 1)
        self.assertEqual(ep.list_now(), 2)


class ReferenceTests(OfflineCase):
    def reference_episode(self):
        ep = episode();ep.items[0].musts = ['finish: different from Maker A equipment with texture rough']
        ep.refs = [(1, ep.items[0].musts[0])]
        ep.rows = {'anchor': {'id': 'anchor', 'seller': 'Maker A', 'attributes': {'texture': 'rough', 'finish': 'satin'}}}
        ep.full = dict(ep.rows);ep.order = ['anchor'];ep.opened = {'anchor'}
        state = ep.reference_state(*ep.refs[0])
        described = {'complete': True, 'conditions': [
            {'kind': 'identity', 'property': 'seller', 'value': 'Maker A', 'source': 'Maker A'},
            {'kind': 'attribute', 'property': 'texture', 'value': 'rough', 'source': 'texture rough'}],
            'operator': 'different', 'relation_source': 'different from'}
        self.assertTrue(ep.describe_reference(state, described))
        return ep, state

    def citation(self, ident='anchor', key='finish', value='satin', scope='attribute'):
        return {'id': ident, 'scope': scope, 'key': key, 'value': value}

    def accepted(self, **extra):
        return {'id': 'anchor', 'source': 'Maker A', 'locator_complete': True,
                'selectors': [{'selector': 0, 'evidence': [self.citation(key='seller', value='Maker A', scope='listing')]},
                              {'selector': 1, 'evidence': [self.citation(key='texture', value='rough')]}],
                'next': None, 'terminal': {'no_onward': True, 'value': 'satin', 'evidence': [self.citation()],
                                         'operator': 'different', 'relation_source': 'different from'}, **extra}

    def test_terminal_reference_requires_real_opened_evidence_and_preserves_negative_relation(self):
        ep, state = self.reference_episode()
        self.assertTrue(ep.accept_reference(state, self.accepted()))
        self.assertEqual(ep.pending_refs(), [])
        self.assertEqual(ep.ref_values[state['need']], 'not satin')

    def test_missing_or_placeholder_cached_value_never_suppresses_pending_reference(self):
        ep, state = self.reference_episode();ep.ref_values[state['need']] = 'whatever the listing shows'
        self.assertEqual(ep.pending_refs(), ep.refs)
        self.assertNotIn(state['need'], ep.ref_values)

    def test_exhausted_counters_cannot_report_unresolved_chain_as_complete(self):
        ep, state = self.reference_episode();ep.chain_rounds = 999;ep.chain_holds = 999;ep.reference_decisions = 36
        with patch.object(ep, 'evidence_decision') as model:
            text, hold = ep.trail()
        self.assertTrue(hold);self.assertIn('unresolved', text);model.assert_not_called()

    def test_wrong_same_seller_product_cannot_ignore_selector_contract(self):
        ep, state = self.reference_episode()
        answer = self.accepted();answer['selectors'] = answer['selectors'][:1]
        self.assertFalse(ep.accept_reference(state, answer))
        ep.rows['anchor']['attributes']['texture'] = 'smooth'
        answer = self.accepted();answer['selectors'][1]['evidence'][0]['value'] = 'smooth'
        self.assertFalse(ep.accept_reference(state, answer))

    def test_unread_and_invented_terminal_values_are_rejected(self):
        ep, state = self.reference_episode();ep.opened.clear()
        self.assertFalse(ep.accept_reference(state, self.accepted()))
        ep.opened.add('anchor');answer = self.accepted();answer['terminal']['value'] = 'polished'
        self.assertFalse(ep.accept_reference(state, answer))

    def test_source_directed_chain_must_open_terminal_page(self):
        ep, state = self.reference_episode()
        quote = 'For finish, use Maker B equipment with texture smooth.'
        ep.rows['anchor']['note'] = quote
        next_description = {'property': 'finish', 'source': quote, 'conditions': [
            {'kind': 'identity', 'property': 'seller', 'value': 'Maker B', 'source': 'Maker B'},
            {'kind': 'attribute', 'property': 'texture', 'value': 'smooth', 'source': 'texture smooth'}]}
        self.assertTrue(ep.accept_reference(state, self.accepted(next=next_description, terminal=None)))
        self.assertTrue(ep.pending_refs());self.assertEqual(len(state['path']), 1)
        ep.rows['terminal'] = {'id': 'terminal', 'seller': 'Maker B', 'attributes': {'texture': 'smooth', 'finish': 'matte'}}
        ep.full['terminal'] = ep.rows['terminal'];ep.opened.add('terminal')
        answer = {'id': 'terminal', 'source': 'Maker B', 'locator_complete': True,
                  'selectors': [{'selector': 0, 'evidence': [self.citation('terminal', 'seller', 'Maker B', 'listing')]},
                                {'selector': 1, 'evidence': [self.citation('terminal', 'texture', 'smooth')]}],
                  'next': None, 'terminal': {'no_onward': True, 'value': 'matte', 'evidence': [self.citation('terminal', value='matte')],
                                           'operator': 'different', 'relation_source': 'different from'}}
        self.assertTrue(ep.accept_reference(state, answer));self.assertEqual(ep.pending_refs(), [])
        self.assertEqual(ep.ref_values[state['need']], 'not matte')

    def test_same_property_onward_direction_cannot_be_treated_as_terminal(self):
        ep, state = self.reference_episode();ep.rows['anchor']['note'] = 'For finish, look to Maker B equipment, not this item.'
        self.assertFalse(ep.accept_reference(state, self.accepted()))

    def test_changed_evidence_invalidates_cached_path_and_restores_target(self):
        ep, state = self.reference_episode();self.assertTrue(ep.accept_reference(state, self.accepted()))
        ep.rows['anchor']['attributes']['finish'] = 'matte'
        self.assertEqual(ep.pending_refs(), ep.refs)
        self.assertFalse(state['complete']);self.assertEqual(state['path'], [])
        self.assertEqual(state['locator'], state['need'])

    def test_withdrawal_removes_reference_and_no_credit_does_not_remove_hold(self):
        ep, state = self.reference_episode();ep.no_credit = True
        self.assertTrue(ep.trail()[1])
        ep.items[0].dropped.append(state['need']);self.assertEqual(ep.pending_refs(), [])

    def test_shared_last_word_does_not_alias_component_scopes(self):
        ep = episode();binding = {'property': 'lens color', 'key': 'frame color', 'same_scope': True, 'reason': 'They match.'}
        self.assertFalse(ep.key_scope_ok('lens color', 'frame color', 1, binding))
        self.assertFalse(ep.key_scope_ok('pack type', 'pack size', 1, binding))

    def test_duplicate_page_cycle_is_rejected_without_mutating_path(self):
        ep, state = self.reference_episode();state['path'].append({'id': 'anchor'})
        self.assertFalse(ep.accept_reference(state, self.accepted()))

    def test_final_action_cannot_bypass_an_unresolved_reference(self):
        ep, state = self.reference_episode();ep.reference_decisions = 36
        with patch.object(agent.ENV_CLIENT, 'post_verbose') as world:
            self.assertIsNone(ep.env('buy', {}))
        world.assert_not_called();self.assertEqual(ep.step, 0)


def proof_episode():
    ep = episode();ep.request = 'One equipment item with a satin surface.'
    ep.items[0].musts = ['surface: satin'];ep.budget = 100
    ep.price_key = 'price'
    ep.rows = {'one': {'id': 'one', 'price': 12, 'category': 'equipment', 'attributes': {'surface': 'satin'}}}
    ep.full = dict(ep.rows);ep.order = ['one'];ep.opened = {'one'};ep.cart_lines = {'one': 1}
    ep.offer_seen = {'one': 0}
    return ep


def cart_proof():
    return {'complete': True, 'source_complete': True,
            'assignments': [{'item': 1, 'id': 'one', 'quantity': 1, 'type_ok': True,
                             'evidence': [{'id': 'one', 'scope': 'listing', 'key': 'category', 'value': 'equipment'}]}],
            'requirements': [{'item': 1, 'need': 'surface: satin', 'status': 'met',
                              'evidence': [{'id': 'one', 'scope': 'attribute', 'key': 'surface', 'value': 'satin'}]}],
            'objective': {'ok': True, 'better': [], 'considered': [], 'reason': 'Any valid item is allowed.'}, 'problems': []}


class DecisionProofTests(OfflineCase):
    def test_fully_grounded_current_cart_is_accepted(self):
        ep = proof_episode();self.assertTrue(ep.validate_cart_proof(cart_proof(), ['one']))

    def test_bare_ok_missing_and_incomplete_reviews_never_approve(self):
        ep = proof_episode()
        for answer in (None, {}, {'ok': True}, {**cart_proof(), 'source_complete': False}):
            self.assertFalse(ep.validate_cart_proof(answer, ['one']))

    def test_wrong_component_and_neighboring_quantity_cannot_prove_requirement(self):
        ep = proof_episode();ep.items[0].musts = ['lens color: clear']
        ep.rows['one']['attributes'] = {'frame color': 'clear'}
        proof = cart_proof();entry = proof['requirements'][0]
        entry.update(need='lens color: clear', evidence=[{'id': 'one', 'scope': 'attribute', 'key': 'frame color', 'value': 'clear'}],
                     binding={'property': 'lens color', 'key': 'frame color', 'same_scope': True, 'reason': 'They look alike.'})
        self.assertFalse(ep.validate_cart_proof(proof, ['one']))
        ep.items[0].musts = ['pack type: individual'];ep.rows['one']['attributes'] = {'pack size': 'individual'}
        entry.update(need='pack type: individual', evidence=[{'id': 'one', 'scope': 'attribute', 'key': 'pack size', 'value': 'individual'}])
        self.assertFalse(ep.validate_cart_proof(proof, ['one']))

    def test_narrow_capability_cannot_pass_as_an_upper_permitted_bound(self):
        ep = proof_episode();need = 'working range: up to +/- 8'
        ep.items[0].musts = [need];ep.need_evidence[(1, need)] = {'relation': 'coverage'}
        ep.rows['one']['attributes'] = {'working range': '0.2-0.5'}
        proof = cart_proof();proof['requirements'] = [{'item': 1, 'need': need, 'status': 'met',
            'evidence': [{'id': 'one', 'scope': 'attribute', 'key': 'working range', 'value': '0.2-0.5'}]}]
        self.assertFalse(ep.validate_cart_proof(proof, ['one']))

    def test_quoted_correct_field_cannot_prove_an_explicitly_different_value(self):
        ep = proof_episode();ep.rows['one']['attributes']['surface'] = 'gloss'
        proof = cart_proof();proof['requirements'][0]['evidence'][0]['value'] = 'gloss'
        self.assertFalse(ep.validate_cart_proof(proof, ['one']))

    def test_quantity_stock_budget_and_unread_replies_invalidate_positive_proof(self):
        for field in ('quantity', 'stock', 'budget', 'reply'):
            ep = proof_episode()
            if field == 'quantity':ep.cart_lines['one'] = 2
            elif field == 'stock':ep.unavail.add('one')
            elif field == 'budget':ep.budget = 10
            else:ep.unread_replies = [(1, 'Question?', 'A changed requirement.')]
            self.assertFalse(ep.validate_cart_proof(cart_proof(), ['one']), field)

    def test_cheapest_proof_must_cover_the_supplied_lower_price_alternative(self):
        ep = proof_episode();ep.brief['pick'] = 'cheapest'
        ep.rows['lower'] = {'id': 'lower', 'price': 8, 'category': 'equipment', 'attributes': {'surface': 'satin'}}
        ep.full['lower'] = ep.rows['lower'];ep.opened.add('lower');ep.order.append('lower')
        proof = cart_proof();self.assertFalse(ep.validate_cart_proof(proof, ['one']))
        proof['objective'].update(considered=['lower'], ok=False, better=['lower'])
        self.assertFalse(ep.validate_cart_proof(proof, ['one']))

    def test_positive_cache_is_reused_and_changed_record_requires_a_new_review(self):
        ep = proof_episode()
        with patch.object(ep, 'evidence_decision', side_effect=[cart_proof(), None]) as model:
            self.assertEqual(ep.cart_check(), '');self.assertEqual(ep.cart_check(), '')
            ep.rows['one']['attributes']['surface'] = 'gloss'
            self.assertTrue(ep.cart_check())
        self.assertEqual(model.call_count, 2)

    def test_unknown_review_can_receive_one_bounded_fresh_adjudication(self):
        ep = proof_episode()
        with patch.object(ep, 'evidence_decision', side_effect=[None, cart_proof()]) as model:
            self.assertTrue(ep.cart_check())
            self.assertTrue(ep.adjudicate_review(['one']))
        self.assertEqual(model.call_count, 2)

    def test_forced_finish_is_not_mistaken_for_a_proposed_decline(self):
        ep = proof_episode();events = []
        def world(path, body):
            events.append(body);return SimpleNamespace(data={'step': body['step'], 'finished': True, 'observation': {'accepted': True}}, error=None)
        with patch.object(ep, 'evidence_decision', return_value=cart_proof()), patch.object(ep, 'decline_check') as decline, patch.object(ep, 'recover', return_value=False), \
             patch.object(agent.ENV_CLIENT, 'post_verbose', side_effect=world), patch.object(agent.time, 'sleep'):
            ep.finish()
        self.assertTrue(ep.finished);self.assertEqual(events[-1]['action']['name'], 'buy');decline.assert_not_called()

    def test_missing_review_cannot_reach_forced_order(self):
        ep = proof_episode()
        with patch.object(ep, 'evidence_decision', return_value=None), patch.object(ep, 'recover', return_value=False), \
             patch.object(agent.ENV_CLIENT, 'post_verbose') as world:
            ep.finish()
        world.assert_not_called();self.assertFalse(ep.finished)

    def test_decline_cannot_claim_an_exact_matching_numeric_candidate_is_wrong(self):
        ep = proof_episode();ep.cart_lines = {};ep.items[0].musts = ['reach: 3 m'];ep.rows['one']['attributes'] = {'reach': '3 m'}
        ep.search_evidence = [{'step': 1, 'tool': 'search', 'args': {'query': 'equipment'}}]
        answer = {'complete': True, 'source_complete': True, 'decline_supported': True, 'fits': [], 'impossible_items': [1],
                  'omitted_assessed': True, 'reason': 'No fit.', 'search_steps': [1], 'checked': [
                      {'id': 'one', 'item': 1, 'need': 'reach: 3 m', 'status': 'mismatch',
                       'evidence': [{'id': 'one', 'scope': 'attribute', 'key': 'reach', 'value': '3 m'}]}]}
        with patch.object(ep, 'evidence_decision', return_value=answer):self.assertTrue(ep.decline_check())

    def test_supported_decline_can_reject_narrow_capability_with_real_evidence(self):
        ep = proof_episode();ep.cart_lines = {};need = 'working range: up to +/- 8'
        ep.items[0].musts = [need];ep.need_evidence[(1, need)] = {'relation': 'coverage'}
        ep.rows['one']['attributes'] = {'working range': '0.2-0.5'}
        ep.search_evidence = [{'step': 1, 'tool': 'search', 'args': {'query': 'equipment'}}]
        answer = {'complete': True, 'source_complete': True, 'decline_supported': True, 'fits': [], 'impossible_items': [1],
                  'omitted_assessed': True, 'reason': 'The observed capability does not cover the requested range.', 'search_steps': [1],
                  'checked': [{'id': 'one', 'item': 1, 'need': need, 'status': 'mismatch',
                               'evidence': [{'id': 'one', 'scope': 'attribute', 'key': 'working range', 'value': '0.2-0.5'}]}]}
        with patch.object(ep, 'evidence_decision', return_value=answer):self.assertEqual(ep.decline_check(), '')

    def test_empty_search_or_unusable_model_reply_does_not_establish_infeasibility(self):
        ep = proof_episode();ep.search_evidence = []
        with patch.object(ep, 'evidence_decision') as model:self.assertTrue(ep.decline_check())
        model.assert_not_called()
        ep.search_evidence = [{'step': 1, 'tool': 'search', 'args': {}}]
        with patch.object(ep, 'evidence_decision', return_value=None):self.assertTrue(ep.decline_check())

    def test_numeric_comparisons_preserve_negative_signs_ranges_and_units(self):
        ep = episode()
        self.assertTrue(ep.scalar_check(1, 'temperature: at least -10 c', '-8 c'))
        self.assertFalse(ep.scalar_check(1, 'temperature: at least -10 c', '-12 c'))
        self.assertFalse(ep.scalar_check(1, 'reach: at most 8 m', '6-10 m'))
        self.assertIsNone(ep.scalar_check(1, 'reach: at most 8 m', '400 mm'))
        self.assertTrue(ep.scalar_check(1, 'reach: 3m', '3 m'))

    def test_exclusion_and_range_do_not_redundantly_settle_the_whole_topic(self):
        ep = episode();ep.items[0].musts = ['surface: not gloss', 'reach: at most 8 m']
        self.assertFalse(ep.topic_settled(1, 'surface'));self.assertFalse(ep.topic_settled(1, 'reach'))


if __name__ == '__main__':
    unittest.main()
