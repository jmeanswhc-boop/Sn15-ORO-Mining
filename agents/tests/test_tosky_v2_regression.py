"""Adversarial state, source-policy and complete-allocation regressions."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
from types import SimpleNamespace
from unittest.mock import patch

from test_tosky_v2 import agent, episode, proof_episode, cart_proof, Client, response, OfflineCase, ROOT


class ChangeAndRecoveryTests(OfflineCase):
    def test_empty_field_values_cannot_establish_requirement_or_reference_evidence(self):
        ep = proof_episode();need = 'surface: satin'
        ep.need_evidence[(1, need)] = {'relation': 'coverage'}
        for value in (None, '', '   ', [], {}):
            ep.rows['one']['attributes']['surface'] = value
            proof = cart_proof();proof['requirements'][0]['evidence'][0]['value'] = value
            self.assertFalse(ep.validate_cart_proof(proof, ['one']))

    def test_new_cheaper_evidence_invalidates_a_positive_optimality_cache(self):
        ep = proof_episode();ep.brief['pick'] = 'cheapest'
        with patch.object(ep, 'evidence_decision', side_effect=[cart_proof(), None]) as model:
            self.assertEqual(ep.cart_check(), '')
            ep.rows['lower'] = {'id': 'lower', 'category': 'equipment', 'price': 8, 'attributes': {'surface': 'satin'}}
            ep.full['lower'] = ep.rows['lower'];ep.opened.add('lower');ep.order.append('lower')
            self.assertTrue(ep.cart_check())
        self.assertEqual(model.call_count, 2)
        proof = cart_proof();proof['objective']['considered'] = ['lower']
        self.assertFalse(ep.validate_cart_proof(proof, ['one']))

    def test_explicit_failed_action_is_not_a_procedure_or_selection_receipt(self):
        ep = episode();ep.roles['cart'] = 'basket';ep.rows = {'one': {'id': 'one'}}
        ep.schemas['basket'] = {'type': 'object', 'properties': {'id': {'type': 'string'}, 'quantity': {'type': 'integer'}}}
        def world(path, body):
            return SimpleNamespace(data={'step': body['step'], 'finished': False, 'observation': {'applied': False}}, error=None)
        with patch.object(agent.ENV_CLIENT, 'post_verbose', side_effect=world), patch.object(agent.time, 'sleep'):
            ep.env('basket', {'id': 'one', 'quantity': 1})
        self.assertIsNone(ep.procedure_first_selection);self.assertNotIn(ep.step, ep.procedure_events)

    def test_repeated_local_holds_handoff_to_repair_instead_of_repeating_full_history(self):
        ep = episode();ep.stalled_dispatches = 2
        with patch.object(ep, 'recover', return_value=False) as repair, patch.object(agent, 'model_client') as model:
            self.assertIsNone(ep.think([{'role': 'user', 'content': 'Unchanged conversation.'}], [agent.LIGHT], 2000, ep.tools))
            self.assertIsNone(ep.think([{'role': 'user', 'content': 'Unchanged conversation.'}], [agent.LIGHT], 2000, ep.tools))
        self.assertEqual(repair.call_count, 1);model.assert_not_called();self.assertFalse(ep.no_credit)

    def test_nonnumeric_and_nonfinite_prices_cannot_approve_a_cart(self):
        ep = proof_episode()
        for value in (float('nan'), float('inf'), -1):
            ep.rows['one']['price'] = value
            self.assertFalse(ep.validate_cart_proof(cart_proof(), ['one']))

    def test_listing_citation_cannot_bypass_missing_property_evidence_with_price(self):
        ep = proof_episode();proof = cart_proof()
        proof['requirements'][0]['evidence'] = [{'id': 'one', 'scope': 'listing', 'key': 'price', 'value': 12}]
        self.assertFalse(ep.validate_cart_proof(proof, ['one']))
        ep.rows['one']['description'] = 'This item is not satin.'
        proof['requirements'][0]['evidence'] = [{'id': 'one', 'scope': 'listing', 'key': 'description', 'value': ep.rows['one']['description']}]
        self.assertFalse(ep.validate_cart_proof(proof, ['one']))

    def test_same_reply_replacement_withdraws_old_value_before_adding_new(self):
        ep = episode();ep.items[0].musts = ['surface: satin'];ep.source_steps['Use matte instead.'] = 2
        reading = {'drop': [{'item': 1, 'property': 'surface', 'source': 'Use matte instead.'}],
                   'needs': [{'item': 1, 'need': 'surface: matte', 'source': 'Use matte instead.'}]}
        changed = [];ep.fold_reading(reading, changed)
        self.assertIn('surface: satin', ep.items[0].dropped)
        self.assertIn('surface: matte', ep.live_of(ep.items[0]));self.assertNotIn('surface: satin', ep.live_of(ep.items[0]))

    def test_later_reaffirmation_can_restore_a_value_but_stale_reply_cannot(self):
        ep = episode();ep.items[0].musts = ['surface: satin'];ep.items[0].dropped = ['surface: satin']
        ep.binding_steps[(1, ep.property_key('surface'))] = 2
        ep.source_steps.update({'Satin originally.': 1, 'Return to satin.': 4})
        ep.fold_reading({'needs': [{'item': 1, 'need': 'surface: satin', 'source': 'Satin originally.'}]}, [])
        self.assertNotIn('surface: satin', ep.live_of(ep.items[0]))
        changed = [];ep.fold_reading({'needs': [{'item': 1, 'need': 'surface: satin', 'source': 'Return to satin.'}]}, changed)
        self.assertEqual(ep.live_of(ep.items[0]), ['surface: satin']);self.assertTrue(changed)

    def test_notice_replacement_is_source_checked_and_keeps_new_value_active(self):
        ep = episode();ep.items[0].musts = ['surface: satin'];ep.step = 3
        line = 'Replace satin with matte.'
        notice = {'kind': 'change', 'budget': None, 'add': [{'item': 1, 'property': 'surface', 'value': 'matte', 'operation': 'eq', 'source': line}],
                  'drop': [{'item': 1, 'need': 'surface: satin', 'source': line}], 'gone': [], 'sold_out': False,
                  'price_moved': False, 'summary': 'The required surface is now matte.'}
        with patch.object(ep, 'json_ask', return_value=notice):self.assertTrue(ep.read_note([line]))
        self.assertIn('surface: matte', ep.live_of(ep.items[0]));self.assertNotIn('surface: satin', ep.live_of(ep.items[0]))
        self.assertIn(line, ep.notice_evidence)

    def test_unquoted_notice_addition_remains_pending(self):
        ep = episode();line = 'Replace satin with matte.'
        notice = {'kind': 'change', 'budget': None, 'add': [{'item': 1, 'need': 'surface: matte'}], 'drop': [],
                  'gone': [], 'sold_out': False, 'price_moved': False, 'summary': 'Change.'}
        with patch.object(ep, 'json_ask', return_value=notice):self.assertEqual(ep.read_note([line]), '')
        self.assertEqual(ep.pending_notices, [line])

    def test_unsupported_reasoning_retries_without_parameter_inside_same_accounting(self):
        ep = episode();client = Client([SimpleNamespace(data=None, error={'status': 422}), response()])
        with patch.object(agent, 'model_client', return_value=client):
            self.assertIsNotNone(ep.evidence_decision('proof', 'Review actual evidence.'))
        self.assertEqual(ep.calls, 2);self.assertIn('reasoning', client.posts[0][1]);self.assertNotIn('reasoning', client.posts[1][1])
        self.assertGreater(ep.inference_budget.uncertain, 0)

    def test_unavailable_initial_interpretation_has_bounded_retries(self):
        ep = episode();ep.brief_decisions = 3
        with patch.object(agent, 'model_client') as client:
            with self.assertRaises(RuntimeError):ep.json_ask(agent.BRIEF_PROMPT, [agent.LIGHT], 4000)
        client.assert_not_called()

    def test_malformed_citations_and_proof_shapes_return_unknown_without_crashing(self):
        ep = proof_episode()
        for value in (None, True, [], {}, 1):
            self.assertFalse(ep.citations_valid([{'id': value, 'scope': 'attribute', 'key': 'surface', 'value': 'satin'}], {'one'}))
            proof = cart_proof();proof['assignments'][0]['id'] = value
            self.assertFalse(ep.validate_cart_proof(proof, ['one']))
        proof = cart_proof();proof['requirements'][0]['need'] = []
        self.assertFalse(ep.validate_cart_proof(proof, ['one']))


class AllocationAndPolicyTests(OfflineCase):
    def test_ninety_isolated_ledgers_account_all_physical_failures_and_missing_usage(self):
        local = threading.local()
        class Workload:
            def __init__(self, number):self.number, self.posts = number, 0
            def post_verbose(self, path, body):
                self.posts += 1
                if self.number % 4 == 0:return SimpleNamespace(data=None, error={'status': 500})
                usage = {} if self.number % 4 == 1 else {'cost': .0001}
                return SimpleNamespace(data={'usage': usage, 'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}, error=None)
        def one(number):
            ep = episode();local.client = Workload(number)
            for _ in range(45):
                if ep.no_credit or ep.calls >= agent.CALL_CAP:break
                ep.evidence_decision('proof', 'Synthetic current-evidence request.', 1600)
            budget = ep.inference_budget
            self.assertEqual(budget.attempts, local.client.posts);self.assertEqual(budget.pending, 0)
            self.assertLessEqual(budget.actual+budget.uncertain, budget.limit)
            return budget.actual+budget.uncertain
        with patch.object(agent, 'model_client', side_effect=lambda _room: local.client), patch.object(agent.time, 'sleep'):
            with ThreadPoolExecutor(max_workers=6) as pool:total = sum(pool.map(one, range(90)))
        self.assertLessEqual(total, 3150000000)

    def test_existing_base_methods_are_unchanged_and_no_forbidden_runtime_access_added(self):
        old = ast.parse((ROOT/'agents/ICDI_chidori_v4.py').read_text());new = ast.parse((ROOT/'agents/Tosky_v2.py').read_text())
        def methods(tree):
            return {(c.name, f.name): ast.dump(f, include_attributes=False) for c in tree.body if isinstance(c, ast.ClassDef)
                    for f in c.body if isinstance(f, ast.FunctionDef)}
        before, after = methods(old), methods(new)
        self.assertTrue(all(after[key] == value for key, value in before.items()))
        allowed = {'json', 'decimal', 'random', 're', 'time', 'typing', 'src.agent.proxy_client'}
        forbidden = {'eval', 'exec', '__import__', 'open', 'write_text', 'write_bytes', 'unlink', 'chmod', 'getenv'}
        for node in ast.walk(new):
            if isinstance(node, ast.Import):self.assertTrue(all(a.name in allowed for a in node.names))
            if isinstance(node, ast.ImportFrom):self.assertIn(node.module, allowed)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertIsNone(re.search(r'\b(?:t-[0-9a-f]{10}|p-[0-9a-f]{12})\b', node.value))
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
                self.assertNotIn(name, forbidden)
        self.assertLess((ROOT/'agents/Tosky_v2.py').stat().st_size, 1000000)


class PublicFailureSnapshotTests(OfflineCase):
    def test_released_reference_chain_directions_are_not_mistaken_for_terminal_values(self):
        path = ROOT/'logs/research/tosky-race-022-2026-11-13/race-feedback/tosky/4c3b4c41-f928-4971-8684-5d801c41a6f1.json'
        if not path.exists():self.skipTest('Public research fixtures are stored separately from submission code.')
        feedback = json.loads(path.read_text())['data'];ep = episode();checks = 0
        def records(value):
            if isinstance(value, dict):
                if value.get('catalog_item_id') and 'attributes' in value:yield value
                else:
                    for entry in value.values():yield from records(entry)
            elif isinstance(value, list):
                for entry in value:yield from records(entry)
        for turn in feedback['interaction']:
            for call in turn['calls']:
                for record in records(call['observation']):
                    note = record.get('listing_notes') or ''
                    if 'For style' in note or 'Its style should' in note:
                        ident = record['catalog_item_id'];ep.rows[ident] = record
                        self.assertTrue(ep.onward_for(ident, 'style: a referenced value'), note);checks += 1
        self.assertGreaterEqual(checks, 4)

    def test_all_six_have_released_passing_controls_from_both_selected_nonchampions(self):
        path = ROOT/'logs/research/tosky-v2-2026-11-13/six-task-public-outcomes.json'
        if not path.exists():self.skipTest('Public research fixtures are stored separately.')
        results = json.loads(path.read_text());self.assertEqual(len(results), 6)
        self.assertTrue(all(row['PRIMO'] == row['Four_Leaf'] == 2 for row in results.values()))
