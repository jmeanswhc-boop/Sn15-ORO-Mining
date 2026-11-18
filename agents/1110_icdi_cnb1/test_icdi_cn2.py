"""Regression tests for the versioned task-update and recovery changes."""
import ast
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_purchase_review as fixtures

agent = fixtures.load('icdi_cn2', fixtures.HERE / 'ICDI_cn2.py')


def scene():
    return fixtures.scene(agent)


class VersionTwoTests(unittest.TestCase):
    def test_original_is_unchanged(self):
        self.assertEqual(hashlib.sha256((fixtures.HERE / 'ICDI_cn1.py').read_bytes()).hexdigest(),
                         'c6ad653288dc14e03bf34d716976f0f17ae4584c0f09dfde24132eb930da3308')

    def test_task_withdrawal_reaches_review_and_invalidates_cache(self):
        episode = scene()
        before = episode.check_key(episode.cart_ids())
        episode.ask_json = Mock(return_value={
            'kind': 'change', 'drop': [{'item': 1, 'need': episode.items[0].musts[0]}],
            'summary': 'The operating interval requirement was withdrawn.'})
        update = 'Please disregard the operating interval requirement now.'
        episode.take_note([update])
        self.assertEqual(episode.needs_now(), [])
        self.assertNotEqual(before, episode.check_key(episode.cart_ids()))
        episode.ask_json = Mock(return_value=fixtures.review_for(episode))
        self.assertEqual(episode.check_cart(), '')
        prompt = episode.ask_json.call_args.args[0]
        self.assertIn(update, prompt)
        self.assertIn('WITHDRAWN REQUIREMENTS', prompt)
        self.assertIn(episode.items[0].musts[0], prompt)

    def test_disjunction_survives_all_cart_category_checks(self):
        episode = scene()
        episode.items = [agent.Item('cases or enclosures', 'Cases & Enclosures', 1, [])]
        episode.rows['sample-1']['category'] = 'Enclosures'
        episode.rows['sample-2'] = {'id': 'sample-2', 'category': 'Cases & Enclosures'}
        self.assertEqual(episode.line_faults(), [])

    def test_non_alternative_retains_fuller_category_check(self):
        episode = scene()
        episode.items = [agent.Item('travel enclosures', '', 1, [])]
        episode.rows['sample-2'] = {'id': 'sample-2', 'category': 'Travel Enclosures'}
        self.assertTrue(any('names item' in error for error in episode.line_faults()))

    def test_rejected_checkout_hidden_until_evidence_changes(self):
        episode = scene()
        episode.ask_json = Mock(return_value=fixtures.review_for(episode, False))
        episode.check_cart()
        self.assertNotIn('purchase', [t['function']['name'] for t in episode.offered()])
        self.assertIn('decline', [t['function']['name'] for t in episode.offered()])
        episode.task_updates.append('An updated specification is available.')
        self.assertIn('purchase', [t['function']['name'] for t in episode.offered()])

    def test_negative_review_does_not_buy_another_model_verdict(self):
        episode = scene()
        episode.ask_json = Mock(return_value=fixtures.review_for(episode, False))
        self.assertTrue(episode.check_cart())
        episode.ask_json.assert_called_once()
        self.assertEqual(episode.ask_json.call_args.args[1], [agent.LIGHT])

    def test_missing_review_can_fall_back(self):
        episode = scene()
        episode.ask_json = Mock(side_effect=[None, fixtures.review_for(episode)])
        self.assertEqual(episode.check_cart(), '')
        self.assertEqual(episode.ask_json.call_count, 2)

    def test_finalization_declines_when_review_blocks_purchase(self):
        episode = scene()
        episode.step = episode.max_steps - 1
        episode.ask_json = Mock(return_value=fixtures.review_for(episode, False))
        actions = []
        def backend(path, body):
            actions.append(body['action']['name'])
            return SimpleNamespace(data={'finished': True, 'observation': {}}, error=None)
        with patch.object(agent._env, 'post_verbose', side_effect=backend):
            episode.finish()
        self.assertEqual(actions, ['decline'])
        self.assertTrue(episode.finished)

    def test_finalization_still_purchases_approved_cart(self):
        episode = scene()
        episode.step = episode.max_steps - 1
        episode.ask_json = Mock(return_value=fixtures.review_for(episode))
        actions = []
        def backend(path, body):
            actions.append(body['action']['name'])
            return SimpleNamespace(data={'finished': True, 'observation': {}}, error=None)
        with patch.object(agent._env, 'post_verbose', side_effect=backend):
            episode.finish()
        self.assertEqual(actions, ['purchase'])
        self.assertTrue(episode.finished)

    def test_budget_exhaustion_stops_inference_without_retry_waits(self):
        episode = scene()
        episode.cost = agent.SPEND_HARD
        with patch.object(agent, '_llm_within') as llm, patch.object(agent.time, 'sleep') as sleep:
            self.assertIsNone(episode.think([], [agent.LIGHT, agent.HEAVY], 100))
        llm.assert_not_called()
        sleep.assert_not_called()

    def test_stalled_actions_stop_after_three_turns(self):
        episode = scene()
        episode.ready_to_ask = Mock(return_value=False)
        episode.asked = True
        episode.chat = []
        episode.think = Mock(return_value={'tool_calls': [{'id': 'x', 'function': {
            'name': 'purchase', 'arguments': '{}'}}]})
        episode.run_calls = Mock(return_value=[('x', 'The evidence is unchanged.')])
        episode.turns(20, agent.SPEND_HARD)
        self.assertEqual(episode.think.call_count, 3)
        self.assertEqual(episode.trace[-1]['kind'], 'stalled')

    def test_optimization_needs_a_source_quote(self):
        for quote, expected in [('', 'any'), ('not in the request', 'any'), ('the cheapest', 'cheapest')]:
            with self.subTest(quote=quote):
                episode = scene()
                episode.items = []
                episode.request = 'Please buy the cheapest enclosure.'
                episode.ask_json = Mock(return_value={'items': [{'name': 'enclosure'}],
                    'roles': {'buy': 'purchase'}, 'pick': 'cheapest', 'pick_quote': quote})
                episode.unmix = Mock()
                episode.read_request()
                self.assertEqual(episode.brief['pick'], expected)

    def test_replacements_keep_older_fully_read_candidates(self):
        episode = scene()
        episode.rows['older-candidate'] = {'id': 'older-candidate', 'category': 'enclosures'}
        episode.full['older-candidate'] = dict(episode.rows['older-candidate'])
        episode.seen_order = ['older-candidate', 'sample-1']
        for index in range(30):
            ident = f'unread-{index}'
            episode.rows[ident] = {'id': ident, 'category': 'enclosures'}
            episode.seen_order.append(ident)
        episode.ask_json = Mock(return_value={'fits': [{'id': 'older-candidate', 'price': 7}]})
        self.assertEqual(episode.replacements(['sample-1']), ['older-candidate'])
        self.assertIn('older-candidate', episode.ask_json.call_args.args[0])

    def test_finalization_with_exhausted_budget_uses_no_model(self):
        episode = scene()
        episode.step = episode.max_steps - 1
        episode.cost = agent.SPEND_HARD + 0.001
        episode.ask_json = Mock(side_effect=AssertionError('No inference budget remains'))
        with patch.object(agent._env, 'post_verbose', return_value=SimpleNamespace(
            data={'finished': True, 'observation': {}}, error=None
        )) as backend:
            episode.finish()
        self.assertEqual(backend.call_args.args[1]['action']['name'], 'decline')
        episode.ask_json.assert_not_called()

    def test_source_is_valid_and_has_no_new_imports(self):
        old = ast.parse((fixtures.HERE / 'ICDI_cn1.py').read_text())
        new = ast.parse((fixtures.HERE / 'ICDI_cn2.py').read_text())
        imports = lambda tree: [ast.dump(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports(old), imports(new))
        self.assertLess((fixtures.HERE / 'ICDI_cn2.py').stat().st_size, 1_000_000)


if __name__ == '__main__':
    unittest.main()
