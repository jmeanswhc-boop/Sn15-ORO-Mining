"""Category-alternative regressions for ICDI_cn4, including observed cart state."""
import ast
import copy
import hashlib
import json
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_purchase_review as fixtures

previous = fixtures.load('cn3_category_baseline', fixtures.HERE / 'ICDI_cn3.py')
agent = fixtures.load('cn4_category_candidate', fixtures.HERE / 'ICDI_cn4.py')


def scene(module=agent, name='cases or enclosures', shelf=''):
    episode = fixtures.scene(module)
    episode.items[0].name = name
    episode.items[0].shelf = shelf
    episode.request = 'Buy one ' + name + ' supporting the required operating interval.'
    episode.rows['sample-2'] = {'id': 'sample-2', 'category': 'Cases & Enclosures'}
    return episode


class CategoryAlternativesTests(unittest.TestCase):
    def test_reproduces_and_fixes_fuller_label_rejection(self):
        old = scene(previous)
        new = scene()
        self.assertTrue(any('names item' in error for error in old.line_faults()))
        self.assertEqual(new.line_faults(), [])

    def test_combined_shelf_does_not_reintroduce_rejection(self):
        for shelf in ['Cases & Enclosures', 'cases or enclosures']:
            with self.subTest(shelf=shelf):
                episode = scene(shelf=shelf)
                episode.rows['sample-2']['category'] = shelf
                self.assertEqual(episode.fuller_label(1, 'enclosures'), '')
                self.assertEqual(episode.named_label(1, 'enclosures'), '')
                self.assertEqual(episode.line_faults(), [])

    def test_separately_required_department_is_still_enforced(self):
        episode = scene(shelf='Travel Equipment')
        episode.rows['sample-3'] = {'id': 'sample-3', 'category': 'Travel Equipment'}
        self.assertEqual(episode.named_label(1, 'enclosures'), 'Travel Equipment')
        self.assertTrue(any('Travel Equipment' in error for error in episode.line_faults()))

    def test_case_spacing_plural_and_multiple_alternatives(self):
        for name, label in [('Cases OR Enclosures', 'Enclosure'),
                            ('cases  or   enclosures', 'Cases'),
                            ('cases or crates or enclosures', 'Crates'),
                            ('travel cases or protective enclosures', 'Protective Enclosures')]:
            with self.subTest(name=name, label=label):
                self.assertTrue(scene(name=name).alternative_label(1, label))

    def test_partial_or_unrelated_categories_do_not_match(self):
        for name, label in [('travel cases or protective enclosures', 'Enclosures'),
                            ('cases or enclosures', 'Enclosure Accessories'),
                            ('cases or enclosures', 'Clocks'),
                            ('cases and enclosures', 'Enclosures'),
                            ('cases or enclosures', ''),
                            ('cases or', 'Cases')]:
            with self.subTest(name=name, label=label):
                self.assertFalse(scene(name=name).alternative_label(1, label))

    def test_non_alternative_fuller_category_check_is_preserved(self):
        for module in [previous, agent]:
            episode = scene(module, name='travel enclosures')
            episode.rows['sample-2']['category'] = 'Travel Enclosures'
            self.assertTrue(any('names item' in error for error in episode.line_faults()))

    def test_quantity_stock_and_budget_checks_still_apply(self):
        episode = scene()
        episode.cart_lines['sample-1'] = 2
        self.assertTrue(any('is held at' in error for error in episode.line_faults()))
        episode.cart_lines['sample-1'] = 1
        episode.unavailable.add('sample-1')
        self.assertTrue(any('sold out' in error for error in episode.line_faults()))
        episode.unavailable.clear()
        episode.budget = 5
        self.assertTrue(any('over the budget' in error for error in episode.line_faults()))

    def test_duplicate_lines_and_unrelated_products_still_fail(self):
        episode = scene()
        episode.rows['sample-3'] = dict(episode.rows['sample-1'], id='sample-3')
        episode.cart_lines['sample-3'] = 1
        self.assertTrue(any('2 lines' in error for error in episode.line_faults()))
        episode.cart_lines = {'sample-1': 1}
        episode.rows['sample-1']['category'] = 'Clocks'
        self.assertTrue(any('not a kind' in error for error in episode.line_faults()))

    def test_checkout_entrypoint_reaches_purchase_after_category_fix(self):
        episode = scene()
        episode.step = 20
        # Keep the surrounding evidence collection deterministic. Category checks,
        # purchase review, and environment transport execute normally.
        for method in ['hunt_cheaper', 'wider_hold', 'reread_offers', 'swap_hint', 'show_links']:
            setattr(episode, method, Mock(return_value=''))
        episode.trail = Mock(return_value=('', False))
        episode.fill_facts = Mock(return_value=[])
        episode.cheaper_rows = Mock(return_value=[])
        episode.ask_json = Mock(return_value=fixtures.review_for(episode))
        with patch.object(agent._env, 'post_verbose', return_value=SimpleNamespace(
            data={'step': 21, 'finished': True, 'observation': {}}, error=None)) as backend:
            episode.run_calls([{'id': 'buy', 'function': {'name': 'purchase', 'arguments': '{}'}}])
        self.assertTrue(episode.finished)
        self.assertEqual(backend.call_args.args[1]['action']['name'], 'purchase')
        self.assertFalse(any(event['kind'] == 'line_hold' for event in episode.trace))

    def test_recorded_category_failure_is_removed(self):
        directory = fixtures.ROOT / 'LV_logs/local-f6809126cbb043e6a3b152a7582d982e'
        results = [json.loads(line)['episode'] for line in (directory / 'episode_results.jsonl').read_text().splitlines()]
        traces = {r['problem_id']: r for r in map(json.loads, (directory / 'sandbox/sandbox_output.jsonl').read_text().splitlines())}
        tested = 0
        for result in results:
            dialogue = traces[result['session_id']]['dialogue']
            brief = next(event['brief'] for event in dialogue if event['kind'] == 'brief')
            if len(brief['items']) != 1 or ' or ' not in brief['items'][0]['name']:
                continue
            faults = [fault for event in dialogue if event['kind'] == 'line_hold'
                      for fault in event['faults'] if 'names item' in fault]
            if not faults:
                continue
            target = re.search(r'p-[a-f0-9]+', faults[0]).group()
            cart_step = next(event['step'] for event in result['ledger']['episode']
                             if event.get('kind') == 'action' and event.get('args', {}).get('chosen_product') == target)
            observed = {}
            def collect(value):
                if isinstance(value, dict):
                    if 'item_pid' in value and 'category' in value:
                        observed[value['item_pid']] = {**value, 'id': value['item_pid']}
                    for child in value.values():
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)
            for event in result['ledger']['episode']:
                if event.get('kind') == 'observation' and event['step'] <= cart_step:
                    collect(event['observation'])
            for module in [previous, agent]:
                episode = fixtures.scene(module)
                item = brief['items'][0]
                episode.items = [module.Item(item['name'], item.get('shelf', ''), 1, item['musts'])]
                episode.rows = copy.deepcopy(observed)
                episode.cart_lines = {target: 1}
                episode.full = {}
                episode.unavailable.clear()
                episode.budget = brief.get('budget')
                category_faults = [fault for fault in episode.line_faults() if 'names item' in fault]
                self.assertEqual(bool(category_faults), module is previous)
            tested += 1
        self.assertGreater(tested, 0)


class PreservationTests(unittest.TestCase):
    def test_cn3_bytes_are_unchanged(self):
        self.assertEqual(hashlib.sha256((fixtures.HERE / 'ICDI_cn3.py').read_bytes()).hexdigest(),
                         '3d59c434838c72ea7e22b71b6a7426ff1746fa84a7dda5f57e02cc7373e11dd0')

    def test_only_category_methods_and_version_docstring_change(self):
        old = ast.parse((fixtures.HERE / 'ICDI_cn3.py').read_text())
        new = ast.parse((fixtures.HERE / 'ICDI_cn4.py').read_text())
        for tree in [old, new]:
            tree.body = tree.body[1:]  # Version docstring.
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == 'Cairn':
                    node.body = [method for method in node.body if not (
                        isinstance(method, ast.FunctionDef)
                        and method.name in {'alternative_label', 'fuller_label', 'named_label'})]
        self.assertEqual(ast.dump(old), ast.dump(new))
        self.assertLess((fixtures.HERE / 'ICDI_cn4.py').stat().st_size, 1_000_000)


if __name__ == '__main__':
    unittest.main()
