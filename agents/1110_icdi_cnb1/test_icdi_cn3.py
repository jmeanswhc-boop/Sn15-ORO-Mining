"""Scoped-update and structured-review regressions for ICDI_cn3."""
import ast
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_purchase_review as fixtures

agent = fixtures.load('icdi_cn3', fixtures.HERE / 'ICDI_cn3.py')


def scene():
    return fixtures.scene(agent)


class ScopedUpdateTests(unittest.TestCase):
    def test_stock_update_keeps_original_cart_identity(self):
        episode = scene()
        episode.ask_json = Mock(return_value={'kind': 'change', 'sold_out': True})
        episode.take_note(['The chosen item has sold out.'], {
            'step': 7, 'tool': 'purchase', 'cart_before': ['sample-1']})
        replacement = copy.deepcopy(episode.rows['sample-1'])
        replacement['id'] = 'replacement'
        episode.rows['replacement'] = replacement
        episode.full['replacement'] = copy.deepcopy(replacement)
        episode.cart_lines = {'replacement': 1}
        episode.ask_json = Mock(return_value=fixtures.review_for(episode))
        self.assertEqual(episode.check_cart(), '')
        event = episode.task_events[0]
        self.assertEqual(event['affected_products'], ['sample-1'])
        self.assertEqual(event['cart_before'], ['sample-1'])
        prompt = episode.ask_json.call_args.args[0]
        self.assertIn('SCOPED TASK UPDATES', prompt)
        self.assertNotIn('The chosen item has sold out.', prompt)
        self.assertNotIn('replacement', event['affected_products'])

    def test_ambiguous_multi_item_stock_is_not_assigned_to_all_products(self):
        episode = scene()
        episode.ask_json = Mock(return_value={'kind': 'change', 'sold_out': True})
        episode.take_note(['A product is no longer available.'], {
            'step': 7, 'tool': 'purchase', 'cart_before': ['sample-1', 'sample-2']})
        self.assertEqual(episode.task_events[0]['affected_products'], [])

    def test_named_stock_event_preserves_named_target(self):
        episode = scene()
        episode.ask_json = Mock(return_value={'kind': 'change', 'gone': ['sample-1']})
        episode.take_note(['The named product sold out.'], {
            'step': 7, 'tool': 'purchase', 'cart_before': ['other']})
        self.assertEqual(episode.task_events[0]['affected_products'], ['sample-1'])

    def test_mixed_stock_and_withdrawal_keep_separate_scopes(self):
        episode = scene()
        need = episode.items[0].musts[0]
        episode.ask_json = Mock(return_value={'kind': 'change', 'sold_out': True,
            'drop': [{'item': 1, 'need': need}], 'budget': 12})
        episode.take_note(['It sold out. The interval is no longer required.'], {
            'step': 7, 'tool': 'purchase', 'cart_before': ['sample-1']})
        self.assertEqual(episode.needs_now(), [])
        self.assertEqual(episode.task_events[0]['budget'], 12)
        self.assertEqual(episode.task_events[0]['affected_products'], ['sample-1'])
        episode.ask_json = Mock(return_value=fixtures.review_for(episode))
        episode.check_cart()
        self.assertIn('WITHDRAWN REQUIREMENTS', episode.ask_json.call_args.args[0])

    def test_act_records_cart_before_mutation(self):
        episode = scene()
        with patch.object(agent._env, 'post_verbose', return_value=SimpleNamespace(
            data={'step': 1, 'finished': False, 'observation': {}}, error=None)):
            episode.act('cart', {'listing': 'sample-1', 'quantity': 0})
        episode.cart_lines.clear()
        self.assertEqual(episode.last_action_context['cart_before'], ['sample-1'])
        self.assertEqual(episode.last_action_context['tool'], 'cart')

    def test_checkout_entrypoint_scopes_refusal_before_replacement(self):
        episode = scene()
        episode.step = episode.max_steps - 2
        episode.swap_hint = Mock(return_value='')
        good = fixtures.review_for(episode)
        episode.ask_json = Mock(side_effect=lambda prompt, *args:
            {'kind': 'change', 'sold_out': True} if prompt.startswith(agent.NOTE) else good)
        with patch.object(agent._env, 'post_verbose', return_value=SimpleNamespace(
            data={'step': episode.step + 1, 'finished': False,
                  'observation': {'refusal': ['The item you selected is now sold out.']}}, error=None)):
            episode.run_calls([{'id': 'checkout', 'function': {
                'name': 'purchase', 'arguments': '{}'}}])
        self.assertEqual(episode.task_events[-1]['affected_products'], ['sample-1'])
        self.assertEqual(episode.task_events[-1]['tool'], 'purchase')

    def test_recorded_checkout_refusals_keep_their_targets(self):
        path = fixtures.ROOT / 'LV_logs/local-b6bf5a6a0d524efaa2832290f5d9aa66/episode_results.jsonl'
        checked = 0
        for line in path.read_text().splitlines():
            episode_data = json.loads(line)['episode']
            for event in episode_data['ledger']['episode']:
                if event.get('kind') != 'finish_check' or event.get('ok') is not False:
                    continue
                order = event.get('order', {})
                if order.get('kind') != 'order' or len(order['lines']) != 1:
                    continue
                ident = order['lines'][0]['id']
                episode = scene()
                episode.rows[ident] = {'id': ident}
                episode.ask_json = Mock(return_value={'kind': 'change', 'sold_out': True})
                episode.take_note(event['observation'], {
                    'tool': 'purchase', 'step': 10, 'cart_before': [ident]})
                self.assertEqual(episode.task_events[-1]['affected_products'], [ident])
                checked += 1
        self.assertGreaterEqual(checked, 20)


class ReviewResponseTests(unittest.TestCase):
    def test_truncation_retries_once_with_more_output_capacity(self):
        episode = scene()
        good = fixtures.review_for(episode)
        sizes = []
        def respond(prompt, models, tokens):
            sizes.append(tokens)
            episode.last_inference_result = {'finish_reason': 'length' if len(sizes) == 1 else 'stop',
                                             'max_tokens': tokens}
            return None if len(sizes) == 1 else good
        episode.ask_json = Mock(side_effect=respond)
        self.assertEqual(episode.check_cart(), '')
        self.assertEqual(sizes, [3000, 6000])
        self.assertEqual(episode.trace[-2]['status'], 'complete')

    def test_incomplete_reviews_are_bounded_across_checkout_retries(self):
        episode = scene()
        def truncated(prompt, models, tokens):
            episode.last_inference_result = {'finish_reason': 'length', 'max_tokens': tokens}
            return None
        episode.ask_json = Mock(side_effect=truncated)
        for _ in range(5):
            self.assertIn('output_limit', episode.check_cart())
        self.assertEqual(episode.ask_json.call_count, 2)
        self.assertEqual(episode.purchase_reviews, {})
        episode.full['sample-1']['attributes']['finish'] = 'new evidence'
        episode.ask_json = Mock(return_value=fixtures.review_for(episode))
        self.assertEqual(episode.check_cart(), '')

    def test_structurally_incomplete_response_is_not_a_product_rejection(self):
        episode = scene()
        episode.ask_json = Mock(return_value={'ok': False, 'problems': ['missing data']})
        self.assertIn('incomplete_review', episode.check_cart())
        self.assertEqual(episode.purchase_reviews, {})
        self.assertEqual(episode.ask_json.call_count, 2)

    def test_complete_negative_review_stays_a_negative_verdict(self):
        episode = scene()
        episode.ask_json = Mock(return_value=fixtures.review_for(episode, False))
        self.assertIn('Purchase held', episode.check_cart())
        self.assertFalse(episode.purchase_reviews[episode.check_key(episode.cart_ids())]['ok'])
        count = episode.ask_json.call_count
        episode.check_cart()
        self.assertEqual(episode.ask_json.call_count, count)

    def test_complete_json_at_limit_is_usable(self):
        episode = scene()
        def respond(*args):
            episode.last_inference_result = {'finish_reason': 'length', 'max_tokens': 3000}
            return fixtures.review_for(episode)
        episode.ask_json = Mock(side_effect=respond)
        self.assertEqual(episode.check_cart(), '')
        episode.ask_json.assert_called_once()

    def test_transport_preserves_finish_reason_and_usage(self):
        episode = scene()
        usage = {'cost': 0.001, 'completion_tokens': 1400,
                 'completion_tokens_details': {'reasoning_tokens': 1399}}
        response = {'choices': [{'finish_reason': 'length', 'message': {'content': ''}}], 'usage': usage}
        proxy = Mock()
        proxy.post_verbose.return_value = SimpleNamespace(data=response, error=None)
        with patch.object(agent, '_llm_within', return_value=proxy):
            self.assertIsNone(episode.ask_json('review', [agent.LIGHT], 1400))
        self.assertEqual(episode.last_inference_result['finish_reason'], 'length')
        self.assertEqual(episode.last_inference_result['usage'], usage)
        self.assertAlmostEqual(episode.cost, 0.001)

    def test_replay_saved_reviews_through_actual_parser(self):
        directory = fixtures.ROOT / 'LV_logs/local-b6bf5a6a0d524efaa2832290f5d9aa66/sandbox'
        total = truncated = unparseable = 0
        for path in directory.glob('request_log_*.jsonl'):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record.get('kind') != 'summary' or record.get('path') != '/inference/chat/completions':
                    continue
                request = record['json_data']
                prompt = request['messages'][0].get('content', '')
                if not isinstance(prompt, str) or not prompt.startswith('Review a proposed purchase'):
                    continue
                total += 1
                choice = (record.get('response', {}).get('choices') or [{}])[0]
                truncated += choice.get('finish_reason') == 'length'
                unparseable += agent._json_obj(choice.get('message', {}).get('content')) is None
        self.assertEqual((total, truncated, unparseable), (88, 38, 37))


class PreservationTests(unittest.TestCase):
    def test_previous_versions_are_unchanged(self):
        hashes = {'ICDI_cn1.py': 'c6ad653288dc14e03bf34d716976f0f17ae4584c0f09dfde24132eb930da3308',
                  'ICDI_cn2.py': '4a2a6faa403f5940dc1b7d68b70777a91e6d7072130b23d9d3ae618b2ea78630'}
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256((fixtures.HERE / name).read_bytes()).hexdigest(), expected)

    def test_baseline_policy_and_unrelated_methods_are_preserved(self):
        old = ast.parse((fixtures.HERE / 'ICDI_cn1.py').read_text())
        new = ast.parse((fixtures.HERE / 'ICDI_cn3.py').read_text())
        def methods(tree):
            return {n.name: ast.dump(n) for c in tree.body if isinstance(c, ast.ClassDef)
                    and c.name == 'Cairn' for n in c.body if isinstance(n, ast.FunctionDef)}
        allowed = {'__init__', 'act', 'take_note', 'check_key', 'check_cart', 'run_calls', 'call_model'}
        a, b = methods(old), methods(new)
        for name in a:
            if name not in allowed:
                self.assertEqual(a[name], b[name], name)
        for name in ['LIGHT', 'HEAVY', 'SPARE', 'SPEND_SOFT', 'SPEND_HARD', 'LOOP_CALLS', 'ALL_CALLS']:
            self.assertEqual(getattr(agent, name), getattr(fixtures.agent, name))
        imports = lambda tree: [ast.dump(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports(old), imports(new))
        self.assertLess((fixtures.HERE / 'ICDI_cn3.py').stat().st_size, 1_000_000)


if __name__ == '__main__':
    unittest.main()
