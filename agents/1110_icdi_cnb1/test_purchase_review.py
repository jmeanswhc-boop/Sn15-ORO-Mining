"""Offline purchase-boundary regressions; all inline shopping data is invented.

Run from the repository root:
python3 -m unittest discover -s agents/1110_icdi_cnb1 -p 'test_*.py' -v
"""

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = ROOT / 'champions/Cairn_v2_06aef3d7-bf0f-779c-8000-ee9b2a46ce76/agent.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = load('icdi_cn1', HERE / 'ICDI_cn1.py')
baseline = load('cairn_v2_baseline', BASELINE)


def tool(name, properties=None, required=None):
    return {'type': 'function', 'function': {
        'name': name, 'description': name,
        'parameters': {'type': 'object', 'properties': properties or {},
                       'required': required or [], 'additionalProperties': False}}}


def scene(module=agent):
    tools = [tool('purchase'), tool('decline'), tool('cart', {
        'listing': {'type': 'string'},
        'quantity': {'type': 'integer', 'minimum': 0}}, ['listing', 'quantity'])]
    episode = module.Cairn({'environment': {
        'binding': {'session_id': 'test-session'},
        'policy_view': {'query': 'Buy one enclosure supporting the entire required operating interval.',
                        'tools': tools, 'max_steps': 38}}})
    episode.env_gap = 0
    episode.roles = {'buy': 'purchase', 'refuse': 'decline', 'cart': 'cart'}
    episode.brief = {'pick': 'any', 'buy_args': {}, 'refuse_args': {}}
    episode.items = [module.Item('enclosure', 'enclosures', 1,
                                ['operating interval: covers the requested interval'])]
    episode.rows = {'sample-1': {
        'id': 'sample-1', 'category': 'enclosures', 'name': 'Portable enclosure',
        'price': 8, 'stock': True,
        'attributes': {'operating interval': 'the complete requested interval',
                       'finish': 'unpainted'}}}
    episode.full = copy.deepcopy(episode.rows)
    episode.seen_order = ['sample-1']
    episode.cart_lines = {'sample-1': 1}
    episode.cart_touched = True
    episode.price_key = 'price'
    episode.stock_key = 'stock'
    episode.id_key = 'id'
    return episode


def review_for(episode, supported=True):
    fields = agent._purchase_fields({i: episode.full[i] for i in episode.cart_ids()})
    checks = []
    for requirement in episode.purchase_requirements():
        field = next(f for f in fields if f['path'] == (
            ['name'] if requirement['text'].startswith('Requested product:')
            else ['attributes', 'operating interval']))
        checks.append({'requirement': requirement['number'], 'listing': field['listing'],
                       'interpretation': requirement['text'],
                       'status': 'supported' if supported else 'contradicted',
                       'evidence': [field['number']],
                       'reason': 'The observed field covers the requested property.'
                       if supported else 'The observed interval does not cover the requested interval.'})
    return {'ok': supported, 'checks': checks, 'problems': [], 'better': []}


def backend():
    requests = []

    def respond(path, body):
        requests.append(copy.deepcopy((path, body)))
        finished = body['action']['name'] == 'purchase'
        return SimpleNamespace(data={'step': body['step'], 'finished': finished,
                                     'observation': {'accepted': True}}, error=None)

    return requests, respond


class ReviewContractTests(unittest.TestCase):
    def setUp(self):
        self.episode = scene()
        self.requirements = self.episode.purchase_requirements()
        self.fields = agent._purchase_fields(self.episode.full)

    def findings(self, review):
        return agent._purchase_findings(review, self.requirements, self.fields, ['sample-1'])

    def test_complete_supported_review_passes(self):
        self.assertTrue(self.findings(review_for(self.episode))['ok'])

    def test_global_yes_cannot_replace_requirement_checks(self):
        self.assertFalse(self.findings({'ok': True})['ok'])

    def test_every_live_requirement_needs_a_check(self):
        review = review_for(self.episode)
        review['checks'].pop()
        self.assertFalse(self.findings(review)['ok'])

    def test_missing_contradicted_or_unknown_evidence_blocks(self):
        for status in ('contradicted', 'unknown', None, True):
            with self.subTest(status=status):
                review = review_for(self.episode)
                review['checks'][-1]['status'] = status
                self.assertFalse(self.findings(review)['ok'])

    def test_nonexistent_empty_and_wrongly_typed_citations_block(self):
        for citations in ([], [99999], [True], ['1'], None):
            with self.subTest(citations=citations):
                review = review_for(self.episode)
                review['checks'][-1]['evidence'] = citations
                self.assertFalse(self.findings(review)['ok'])

    def test_citation_to_another_listing_cannot_support_cart_line(self):
        fields = self.fields + [{'number': 999, 'listing': 'sample-other',
                                 'path': ['attributes', 'operating interval'], 'value': 'fits'}]
        review = review_for(self.episode)
        review['checks'][-1]['evidence'] = [999]
        result = agent._purchase_findings(review, self.requirements, fields, ['sample-1'])
        self.assertFalse(result['ok'])

    def test_multiple_products_cannot_pool_their_properties(self):
        requirements = [{'number': 1, 'item': 1, 'text': 'first property'},
                        {'number': 2, 'item': 1, 'text': 'second property'}]
        fields = [{'number': 1, 'listing': 'left'}, {'number': 2, 'listing': 'right'}]
        checks = [{'requirement': n, 'listing': ident, 'interpretation': 'required',
                   'status': 'supported', 'evidence': [n], 'reason': 'supported'}
                  for n, ident in ((1, 'left'), (2, 'right'))]
        result = agent._purchase_findings({'ok': True, 'checks': checks}, requirements,
                                          fields, ['left', 'right'])
        self.assertFalse(result['ok'])

    def test_duplicate_checks_and_truthy_non_boolean_approval_block(self):
        review = review_for(self.episode)
        review['checks'].append(copy.deepcopy(review['checks'][0]))
        self.assertFalse(self.findings(review)['ok'])
        for approval in ('true', 1, None):
            review = review_for(self.episode)
            review['ok'] = approval
            self.assertFalse(self.findings(review)['ok'])

    def test_fields_preserve_complete_values_and_nested_paths(self):
        long_value = 'endpoint ' + 'x' * 3500
        fields = agent._purchase_fields({'example': {'facts': {'limits': [long_value, 0, False]}}})
        self.assertEqual([f['value'] for f in fields], [long_value, 0, False])
        self.assertEqual(fields[0]['path'], ['facts', 'limits', 0])

    def test_dropped_needs_are_excluded_and_new_needs_are_included(self):
        old = self.episode.items[0].musts[0]
        self.episode.items[0].dropped.append(old)
        self.episode.items[0].told.append('surface: specified by the shopper')
        texts = [r['text'] for r in self.episode.purchase_requirements()]
        self.assertNotIn(old, texts)
        self.assertIn('surface: specified by the shopper', texts)


class PurchaseBoundaryTests(unittest.TestCase):
    def test_reproduces_baseline_veto_bypass(self):
        episode = scene(baseline)
        episode.ask_json = Mock(return_value={'ok': False, 'problems': ['Not supported.']})
        self.assertTrue(episode.check_cart())
        self.assertEqual(episode.check_cart(), '')
        self.assertIs(episode.verdicts[episode.check_key(episode.cart_ids())], False)

    def test_rejection_persists_without_repeated_inference(self):
        episode = scene()
        episode.ask_json = Mock(return_value=review_for(episode, False))
        first = episode.check_cart()
        calls = episode.ask_json.call_count
        self.assertIn('Purchase held', first)
        self.assertEqual(episode.check_cart(), first)
        self.assertEqual(episode.ask_json.call_count, calls)

    def test_independent_review_can_resolve_a_false_alarm(self):
        episode = scene()
        episode.ask_json = Mock(side_effect=[review_for(episode, False), review_for(episode)])
        self.assertEqual(episode.check_cart(), '')
        self.assertEqual([c.args[1] for c in episode.ask_json.call_args_list],
                         [[agent.LIGHT], [agent.HEAVY]])

    def test_reproduces_baseline_forced_finish_bypass(self):
        episode = scene(baseline)
        episode.ask_json = Mock(return_value={'ok': False, 'problems': ['Not supported.']})
        self.assertTrue(episode.check_cart())
        requests, respond = backend()
        with patch.object(baseline._env, 'post_verbose', side_effect=respond):
            episode.finish()
        self.assertEqual([r[1]['action']['name'] for r in requests], ['purchase'])

    def test_acceptance_is_cached_without_repeated_inference(self):
        episode = scene()
        episode.ask_json = Mock(return_value=review_for(episode))
        self.assertEqual(episode.check_cart(), '')
        self.assertEqual(episode.check_cart(), '')
        self.assertEqual(episode.ask_json.call_count, 1)

    def test_changed_input_invalidates_both_positive_and_negative_cache(self):
        changes = [
            lambda e: e.full['sample-1']['attributes'].update({'operating interval': 'a new interval'}),
            lambda e: e.full['sample-1'].update({'price': 10}),
            lambda e: e.cart_lines.update({'sample-1': 2}),
            lambda e: e.items[0].told.append('additional requirement'),
            lambda e: setattr(e, 'budget', 20),
            lambda e: e.qa.append(('clarify', 'The requirement has changed.')),
            lambda e: e.unavailable.add('sample-1'),
        ]
        for supported in (False, True):
            for index, change in enumerate(changes):
                with self.subTest(supported=supported, change=index):
                    episode = scene()
                    episode.ask_json = Mock(return_value=review_for(episode, supported))
                    episode.check_cart()
                    old = episode.check_key(episode.cart_ids())
                    change(episode)
                    self.assertNotEqual(episode.check_key(episode.cart_ids()), old)
                    self.assertNotIn(episode.check_key(episode.cart_ids()), episode.purchase_reviews)
                    episode.ask_json.reset_mock()
                    episode.ask_json.return_value = review_for(episode)
                    self.assertEqual(episode.check_cart(), '')
                    episode.ask_json.assert_called_once()

    def test_final_step_purchase_cannot_bypass_rejected_check(self):
        episode = scene()
        episode.step = episode.max_steps - 1
        episode.ask_json = Mock(return_value=review_for(episode, False))
        with patch.object(agent._env, 'post_verbose') as post:
            replies = episode.run_calls([{'id': 'call', 'function': {
                'name': 'purchase', 'arguments': '{}'}}])
        post.assert_not_called()
        self.assertIn('Purchase held', replies[0][1])
        self.assertEqual(episode.step, episode.max_steps - 1)

    def test_forced_finish_preserves_veto_and_does_not_invent_a_decline(self):
        episode = scene()
        episode.ask_json = Mock(return_value=review_for(episode, False))
        with patch.object(agent._env, 'post_verbose') as post:
            result = episode.finish(again=True)
        post.assert_not_called()
        self.assertIn('Purchase held', result)

    def test_direct_purchase_transport_is_guarded(self):
        episode = scene()
        episode.ask_json = Mock(return_value=review_for(episode, False))
        with patch.object(agent._env, 'post_verbose') as post:
            self.assertIsNone(episode.act('purchase', {}))
        post.assert_not_called()

    def test_valid_purchase_paths_keep_singular_action_and_step_sequence(self):
        for route in ('run_calls', 'finish', 'act'):
            with self.subTest(route=route):
                episode = scene()
                episode.step = episode.max_steps - 1
                episode.ask_json = Mock(return_value=review_for(episode))
                requests, respond = backend()
                with patch.object(agent._env, 'post_verbose', side_effect=respond):
                    if route == 'run_calls':
                        episode.run_calls([{'id': 'call', 'function': {
                            'name': 'purchase', 'arguments': '{}'}}])
                    elif route == 'finish':
                        episode.finish()
                    else:
                        episode.act('purchase', {})
                self.assertEqual(len(requests), 1)
                path, payload = requests[0]
                self.assertEqual(path, '/environment/call')
                self.assertEqual(payload['step'], 38)
                self.assertEqual(payload['action'], {'name': 'purchase', 'args': {}})
                self.assertNotIn('actions', payload)
                self.assertTrue(episode.finished)
                self.assertEqual(episode.ask_json.call_count, 1)

    def test_recovery_purchase_checks_actual_cart_after_planned_cart(self):
        episode = scene()
        approved = review_for(episode)
        episode.cart_lines = {}
        episode.confirmed = 'sample-1'
        episode.lead_items = {'1': ['sample-1']}
        episode.ask_json = Mock(return_value=approved)
        requests, respond = backend()
        with patch.object(agent._env, 'post_verbose', side_effect=respond):
            result = episode.complete_cart()
        self.assertTrue(result)
        self.assertEqual([r[1]['action']['name'] for r in requests], ['cart', 'purchase'])
        self.assertEqual([r[1]['step'] for r in requests], [1, 2])
        self.assertTrue(episode.finished)

    def test_recovery_purchase_cannot_bypass_new_contradiction(self):
        episode = scene()
        approved, rejected = review_for(episode), review_for(episode, False)
        episode.cart_lines = {}
        episode.confirmed = 'sample-1'
        episode.lead_items = {'1': ['sample-1']}
        episode.ask_json = Mock(side_effect=[approved, rejected, rejected])
        requests, respond = backend()
        with patch.object(agent._env, 'post_verbose', side_effect=respond):
            result = episode.complete_cart()
        self.assertIn('Purchase held', result)
        self.assertEqual([r[1]['action']['name'] for r in requests], ['cart'])
        self.assertFalse(episode.finished)

    def test_model_outage_is_retryable_without_buying_or_declining(self):
        episode = scene()
        episode.ask_json = Mock(return_value=None)
        with patch.object(agent._env, 'post_verbose') as post:
            result = episode.finish(again=True)
        post.assert_not_called()
        self.assertIn('unavailable', result)
        self.assertFalse(episode.purchase_reviews)
        episode.ask_json.return_value = review_for(episode)
        self.assertEqual(episode.check_cart(), '')

    def test_empty_cart_cannot_pass_review(self):
        episode = scene()
        episode.cart_lines = {}
        with patch.object(agent._env, 'post_verbose') as post:
            self.assertIsNone(episode.act('purchase', {}))
        post.assert_not_called()

    def test_existing_decline_and_nonpurchase_calls_are_not_reviewed(self):
        for name, args in [('decline', {}), ('cart', {'listing': 'sample-1', 'quantity': 1})]:
            episode = scene()
            episode.ask_json = Mock(side_effect=AssertionError('Unexpected purchase review'))
            requests, respond = backend()
            with patch.object(agent._env, 'post_verbose', side_effect=respond):
                episode.act(name, args)
            self.assertEqual(len(requests), 1)

    def test_shopper_reply_capture_preserves_message_and_step_counts(self):
        episode = scene()
        episode.roles['ask'] = 'ask'
        episode.schemas['ask'] = tool('ask', {'text': {'type': 'string'}}, ['text'])['function']['parameters']
        episode.msg_limit = 3
        reply = SimpleNamespace(data={'step': 1, 'finished': False,
                                     'observation': {'reply': 'The required interval has changed.'}}, error=None)
        with patch.object(agent._env, 'post_verbose', return_value=reply) as post:
            sent = episode.send([(1, 'operating interval', 'Which interval is required?')])
        self.assertEqual(episode.qa, [('Which interval is required?', 'The required interval has changed.')])
        self.assertEqual(len(sent), 1)
        self.assertEqual(episode.msg_used, 1)
        self.assertEqual(episode.step, 1)
        post.assert_called_once()


class RequestPreservationTests(unittest.TestCase):
    def parse(self, request, item):
        episode = scene()
        episode.request = request
        episode.items = []
        brief = {'items': [item], 'roles': {'buy': 'purchase', 'refuse': 'decline'},
                 'wishes': ['striped pattern']}
        episode.ask_json = Mock(return_value=brief)
        episode.unmix = Mock()
        episode.read_request()
        episode.ask_json.assert_called_once()
        return episode

    def test_explicit_property_survives_missing_must_and_optional_wish(self):
        episode = self.parse('I want a jacket with a striped pattern; I will check it myself.', {
            'name': 'jacket', 'musts': [], 'product_quote': 'jacket',
            'explicit_requirements': [{'quote': 'with a striped pattern',
                                       'need': 'pattern: striped'}]})
        self.assertIn((1, 'pattern: striped'), episode.needs_now())
        self.assertIn('pattern: striped', [r['text'] for r in episode.purchase_requirements()])

    def test_allowed_alternative_reaches_category_matching(self):
        episode = self.parse('Please buy trousers or jeans with a loose fit.', {
            'name': 'trousers', 'shelf': 'Trousers', 'musts': ['fit: loose'],
            'product_quote': 'trousers or jeans'})
        self.assertEqual(episode.items[0].name, 'trousers or jeans')
        self.assertTrue(episode.owns(episode.items[0], 'Jeans'))
        self.assertTrue(episode.owns(episode.items[0], 'Trousers'))
        self.assertFalse(episode.owns(episode.items[0], 'Clocks'))

    def test_unquoted_and_malformed_additions_are_ignored(self):
        episode = self.parse('Buy a jacket.', {
            'name': 'jacket', 'product_quote': 'jacket or shoes', 'musts': [],
            'explicit_requirements': [None, {'quote': 'silk', 'need': 'material: silk'},
                                      {'quote': 'jacket', 'need': 'pattern:'}]})
        self.assertEqual(episode.items[0].name, 'jacket')
        self.assertEqual(episode.needs_now(), [])

    def test_explicit_requirement_can_still_be_withdrawn(self):
        episode = self.parse('Buy a green jacket.', {
            'name': 'jacket', 'musts': ['color: green'],
            'explicit_requirements': [{'quote': 'green', 'need': 'color: green'}]})
        self.assertEqual(episode.needs_now(), [(1, 'color: green')])
        episode.take_heard({'drop': [{'item': 1, 'property': 'color'}]}, [])
        self.assertEqual(episode.needs_now(), [])
        self.assertEqual(len(episode.purchase_requirements()), 1)

    def test_old_response_format_remains_supported(self):
        episode = self.parse('Buy a green jacket.', {
            'name': 'jacket', 'musts': ['color: green']})
        self.assertEqual(episode.items[0].name, 'jacket')
        self.assertEqual(episode.needs_now(), [(1, 'color: green')])


class ScopeAndStaticTests(unittest.TestCase):
    def test_existing_prompts_models_and_configuration_are_preserved(self):
        def assignments(path):
            tree = ast.parse(path.read_text())
            return {n.targets[0].id: ast.dump(n) for n in tree.body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        old = assignments(BASELINE)
        new = assignments(HERE / 'ICDI_cn1.py')
        for name, value in old.items():
            if name not in {'CHECK', 'BRIEF'}:
                with self.subTest(name=name):
                    self.assertEqual(new[name], value)

    def test_unrelated_components_have_identical_asts(self):
        old = ast.parse(BASELINE.read_text())
        new = ast.parse((HERE / 'ICDI_cn1.py').read_text())

        def definitions(tree):
            result = {}
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == 'Cairn':
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            result['Cairn.' + method.name] = ast.dump(method)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    result[node.name] = ast.dump(node)
            return result

        allowed = {'Cairn.' + name for name in ('__init__', 'act', 'send', 'check_key',
                   'check_cart', 'complete_cart', 'run_calls', 'finish', 'read_request')}
        old_defs, new_defs = definitions(old), definitions(new)
        for name, definition in old_defs.items():
            if name not in allowed:
                with self.subTest(component=name):
                    self.assertEqual(new_defs[name], definition)

    def test_submission_syntax_encoding_entrypoint_and_prohibited_operations(self):
        source = (HERE / 'ICDI_cn1.py').read_bytes()
        self.assertLess(len(source), 1_000_000)
        tree = ast.parse(source.decode('utf-8'))
        entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'agent_main']
        self.assertEqual(len(entries), 1)
        prohibited = {'os', 'posix', 'nt', 'subprocess', 'commands', 'socket', 'http',
                      'urllib', 'requests', 'ctypes', 'cffi', 'pickle', 'shutil',
                      'base64', 'binascii', 'codecs', 'zlib'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({a.name.split('.')[0] for a in node.names} & prohibited)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or '').split('.')[0], prohibited)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {'eval', 'exec', '__import__', 'open'})
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {'write_text', 'write_bytes', 'unlink',
                                                'symlink_to', 'hardlink_to', 'chmod', 'rmdir'})

    def test_public_task_identifiers_are_not_in_agent_constants(self):
        data = json.loads((ROOT / 'problems/qualifiers/suite-160/api_response.json').read_text())
        task_ids = {t['task_id'] for t in data['qualifying_tasks']}
        tree = ast.parse((HERE / 'ICDI_cn1.py').read_text())
        constants = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                     and isinstance(n.value, str)]
        self.assertFalse({task for task in task_ids for text in constants if task in text})


if __name__ == '__main__':
    unittest.main()
