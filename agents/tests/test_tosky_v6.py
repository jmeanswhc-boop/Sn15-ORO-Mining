"""Offline contracts and falsifying examples; these are not ORO scores."""
from pathlib import Path
import unittest
from unittest.mock import patch

from agents.tests import test_tosky_v4 as v4
from agents.tests import test_tosky_v5 as v5

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'agents/Tosky_v6.py'
original_load = v4.load


def load(path=SOURCE):
    return original_load(path)


def case(name='widget', musts=None):
    ns = load()
    e = v4.episode(ns)
    e.items = [ns['Item'](name, '', 1, musts or [])]
    e.id_key, e.price_key, e.stock_key = 'id', 'price', 'stock'
    e.category_key = lambda: 'category'
    return ns, e


class EvidenceContracts(unittest.TestCase):
    def test_audit_preserves_explicit_component_qualifier_from_quote(self):
        _, e = case('lantern')
        e.request = 'I need a light; the type of light is a lantern.'
        brief = {'items': [{'name': 'lantern', 'musts': []}]}
        e.ask_json = lambda *a: {'add': [{'item': 1, 'kind': 'property', 'property': 'type',
                                        'constraint': 'lantern', 'quote': 'the type of light is a lantern'}]}
        e.audit_request(brief)
        self.assertEqual(brief['items'][0]['musts'], ['type of light: lantern'])

    def test_distinct_dimensions_are_unknown_not_a_false_veto(self):
        _, e = case(musts=['style: formal'])
        e.rows['a'] = {'id': 'a', 'category': 'Widgets', 'price': 20, 'stock': True}
        e.full['a'] = {**e.rows['a'], 'attributes': {'style': 'formal', 'widget style': 'hinged'}}
        self.assertIsNone(e.key_verdict('a', 'style: formal')[0])
        self.assertFalse(e.unfit('a'))
        self.assertFalse(e.proven('a', 1))
        self.assertIn('widget style', '\n'.join(e.key_notes(['a'])))

    def test_conflict_is_not_resolved_by_matching_the_desired_value(self):
        _, e = case()
        e.rows['a'] = {'id': 'a', 'category': 'Widgets'}
        e.full['a'] = {**e.rows['a'], 'attributes': {'finish': 'matte', 'widget finish': 'glossy'}}
        for need in ('finish: matte', 'finish: glossy', 'finish: not matte'):
            self.assertIsNone(e.key_verdict('a', need)[0], need)
        self.assertFalse(e.key_verdict('a', 'widget finish: matte')[0])
        self.assertTrue(e.key_verdict('a', 'widget finish: glossy')[0])

    def test_unambiguous_evidence_exclusions_and_other_components(self):
        _, e = case()
        e.rows['a'] = {'id': 'a', 'category': 'Widgets'}
        e.full['a'] = {**e.rows['a'], 'attributes': {'finish': 'matte', 'handle finish': 'glossy'}}
        self.assertTrue(e.key_verdict('a', 'finish: matte')[0])
        self.assertFalse(e.key_verdict('a', 'finish: not matte')[0])
        self.assertFalse(e.key_verdict('a', 'finish: glossy')[0])
        e.full['a']['attributes']['widget finish'] = 'matte'
        self.assertTrue(e.key_verdict('a', 'finish: matte')[0])

    def test_title_word_does_not_prove_an_explicitly_qualified_property(self):
        _, e = case(musts=['type of widget: ornate'])
        e.rows['a'] = {'id': 'a', 'category': 'Widgets', 'title': 'Ornate widget'}
        e.full['a'] = {**e.rows['a'], 'attributes': {'style': 'ornate', 'material': 'metal'}}
        self.assertIsNone(e.key_verdict('a', 'type of widget: ornate')[0])
        self.assertFalse(e.states_needs('a', 1))

    def test_missing_property_is_not_a_proven_failure_or_success(self):
        _, e = case(musts=['packaging: cans'])
        for ident in ('a', 'b'):
            e.rows[ident] = {'id': ident, 'category': 'Widgets', 'price': 20, 'stock': True}
        e.full['a'] = {**e.rows['a'], 'attributes': {'material': 'metal', 'color': 'blue'}}
        e.full['b'] = {**e.rows['b'], 'attributes': {'packaging': 'boxes', 'color': 'blue'}}
        self.assertIsNone(e.key_verdict('a', 'packaging: cans')[0])
        self.assertFalse(e.unfit('a'))
        self.assertFalse(e.states_needs('a', 1))
        self.assertTrue(e.unfit('b'))

    def test_numeric_conflict_is_uncertain_but_explicit_scope_is_enforced(self):
        _, e = case()
        e.rows['a'] = {'id': 'a', 'category': 'Widgets'}
        e.full['a'] = {**e.rows['a'], 'attributes': {'length': '9 m', 'widget length': '5 m'}}
        self.assertIsNone(e.key_verdict('a', 'length: at most 6 m')[0])
        self.assertFalse(e.figure_conflict('a', 'length: 6 m'))
        self.assertTrue(e.key_verdict('a', 'widget length: at most 6 m')[0])
        self.assertFalse(e.key_verdict('a', 'widget length: at most 4 m')[0])

    def test_uncertain_cheap_candidate_survives_bounded_recovery_context(self):
        _, e = case(musts=['shape: square'])
        e.brief['pick'] = 'cheapest'
        for n in range(30):
            ident = 'candidate-' + str(n)
            e.rows[ident] = {'id': ident, 'category': 'Widgets', 'price': 30+n, 'stock': True}
            e.full[ident] = {**e.rows[ident], 'attributes': {'shape': 'round', 'widget shape': 'square'}}
        e.seen_order = list(e.rows)
        self.assertIn('candidate-0', e.recovery_pool(['candidate-29']))
        self.assertLessEqual(len(e.recovery_pool(['candidate-29'])), 25)
        self.assertFalse(e.proven('candidate-0', 1))

    def test_uncertain_candidates_do_not_displace_proven_replacements(self):
        _, e = case(musts=['finish: matte'])
        e.brief['pick'] = 'cheapest'
        e.rows['proven'] = {'id': 'proven', 'category': 'Widgets', 'price': 100, 'stock': True}
        e.full['proven'] = {**e.rows['proven'], 'attributes': {'finish': 'matte', 'material': 'metal'}}
        for n in range(30):
            ident = 'unresolved-' + str(n)
            e.rows[ident] = {'id': ident, 'category': 'Widgets', 'price': n+1, 'stock': True}
            e.full[ident] = {**e.rows[ident], 'attributes': {'finish': 'matte', 'widget finish': 'glossy'}}
        e.seen_order = list(e.rows)
        pool = e.recovery_pool(['unresolved-29'])
        self.assertIn('proven', pool)
        self.assertIn('unresolved-0', pool)
        self.assertLessEqual(len(pool), 25)


class AlternativeContracts(unittest.TestCase):
    def test_allowed_kind_survives_final_cart_validator(self):
        _, e = case('stool or bench')
        e.rows = {'a': {'id': 'a', 'category': 'Stools', 'price': 20},
                  'b': {'id': 'b', 'category': 'Stools & Benches', 'price': 30}}
        e.cart_lines, e.cart_touched = {'a': 1}, True
        self.assertEqual(e.owner_of('Stools'), 1)
        self.assertEqual(e.owner_of('Benches'), 1)
        self.assertFalse(e.line_faults())
        self.assertEqual(e.owner_of('Sofas'), 0)

    def test_explicit_shelf_and_compound_kind_keep_their_guards(self):
        ns, e = case('stool', [])
        e.items[0].shelf = 'Outdoor Stools'
        e.rows = {'a': {'id': 'a', 'category': 'Stools', 'price': 20},
                  'b': {'id': 'b', 'category': 'Outdoor Stools', 'price': 30}}
        e.cart_lines, e.cart_touched = {'a': 1}, True
        self.assertTrue(e.line_faults())
        e.items = [ns['Item']('stool and table set', '', 1, [])]
        e.owners_seen = {}
        e.rows['b']['category'] = 'Stool and Table Sets'
        self.assertTrue(e.line_faults())


class PreservedContracts(unittest.TestCase):
    def test_request_procedures_recovery_and_scripted_successes(self):
        classes = (v5.RequestCoverageTests, v5.ProcedureReuseTests,
                   v5.RecoveryWindowTests, v5.IntegratedPreservationTests)
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in classes)
        for cls in (v4.BudgetTests, v4.ProcedureTests, v4.ChangeTests, v4.EvidenceTests):
            for test in unittest.defaultTestLoader.loadTestsFromTestCase(cls):
                # V6 deliberately declines a severely shortened expensive-model answer.
                if test._testMethodName != 'test_long_context_tier_and_output_cap_are_budgeted':
                    suite.addTest(test)
        with patch.object(v5, 'load', load), patch.object(v5, 'SOURCE', SOURCE), patch.object(v4, 'load', load):
            result = unittest.TestResult()
            suite.run(result)
        self.assertTrue(result.wasSuccessful(), (result.failures, result.errors))


class BudgetRecoveryContracts(unittest.TestCase):
    def test_ninety_invocations_include_reserved_recovery_spending(self):
        ns = load()
        total = 0.0
        for _ in range(90):
            e = v4.episode(ns)
            v4.transport(ns, [v4.success(None)] * ns['ALL_CALLS'])
            for recovery in (False, True):
                e.in_recovery = recovery
                for _ in range(ns['ALL_CALLS']):
                    _, status = e.call_model(ns['LIGHT'], [], 3000, None)
                    if status == 'budget_denied':
                        break
            self.assertLessEqual(e.cost, ns['EPISODE_USD'] + 1e-12)
            total += e.cost
        self.assertLessEqual(total, 4.5 + 1e-9)
    def test_call_admission_uses_compaction_without_mutating_raw_history(self):
        import json
        ns, e = case()
        e.cost = e.cost_actual = 0.040
        messages = [{'role': 'system', 'content': 'Act on public evidence.'}]
        record = {'id': 'listing-one', 'description': 'listing description ' * 250}
        for i in range(25):
            messages += [{'role': 'assistant', 'tool_calls': [{'id': str(i), 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]},
                         {'role': 'tool', 'tool_call_id': str(i), 'content': json.dumps(record)}]
        before = ns['_dump'](messages)
        sent = v4.transport(ns, [v4.success()])
        answer, _ = e.call_model(ns['LIGHT'], messages, 2000, [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object', 'properties': {}}}}])
        self.assertIsNotNone(answer)
        self.assertEqual(len(sent), 1)
        self.assertEqual(ns['_dump'](messages), before)
        self.assertTrue(any(x.get('kind') == 'context_compacted' for x in e.trace))
    def test_completion_allowance_is_protected_then_available_for_recovery(self):
        ns, e = case()
        body = {'model': ns['LIGHT'], 'messages': [{'role': 'user', 'content': 'Facts'}], 'max_tokens': 2000}
        e.cost = ns['EPISODE_USD'] - ns['FINISH_USD']
        self.assertIsNone(e.quote_inference(body))
        e.in_recovery = True
        quote = e.quote_inference(body)
        self.assertIsNotNone(quote)
        self.assertLessEqual(quote[0] + e.cost, ns['EPISODE_USD'])

    def test_expensive_call_is_not_admitted_with_unusable_output_room(self):
        ns, e = case()
        e.cost = 0.040
        body = {'model': ns['HEAVY'], 'messages': [{'role': 'user', 'content': 'x' * 1000}], 'max_tokens': 3000}
        self.assertIsNone(e.quote_inference(body))
        body['model'] = ns['LIGHT']
        self.assertIsNotNone(e.quote_inference(body))

    def test_length_limited_empty_json_gets_one_priced_fallback(self):
        ns, e = case()
        truncated = {'choices': [{'finish_reason': 'length', 'message': {'content': ''}}],
                     'usage': {'cost': 0.001, 'completion_tokens_details': {'reasoning_tokens': 3500}}}
        sent = v4.transport(ns, [(truncated, None), v4.success(0.0001, '{"ok":true}')])
        self.assertEqual(e.ask_json('Review the observed evidence.', [ns['LIGHT']], 3500), {'ok': True})
        self.assertEqual([b['model'] for _, b in sent], [ns['LIGHT'], ns['SPARE']])
        self.assertAlmostEqual(e.cost, 0.0011)
        self.assertTrue(any(x.get('kind') == 'llm_unusable' for x in e.trace))

    def test_repeated_context_preserves_changed_values_and_tool_protocol(self):
        import json
        ns, e = case()
        old = {'id': 'item-one', 'stock': True, 'price': 30, 'description': 'descriptive text ' * 35}
        new = {**old, 'stock': False, 'price': 31}
        messages = [{'role': 'system', 'content': 'Use public facts.'}, {'role': 'user', 'content': 'Buy carefully.'}]
        for i in range(8):
            messages.append({'role': 'assistant', 'content': None, 'tool_calls': [{'id': str(i), 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]})
            messages.append({'role': 'tool', 'tool_call_id': str(i), 'content': json.dumps({'rows': [new if i == 7 else old]}) + '\n' + 'unchanged requirement ' * 15})
        compact, info = e.compact_messages(messages)
        self.assertLess(len(ns['_dump'](compact)), len(ns['_dump'](messages)))
        marker = info['marker']
        table = json.loads(compact[-1]['content'].rsplit('\nRepeated observation table ', 1)[1].split('\n', 1)[1])
        def decode(node):
            if isinstance(node, dict):
                if set(node) == {marker}:
                    return decode(table[node[marker]])
                return {k: decode(v) for k, v in node.items()}
            return [decode(x) for x in node] if isinstance(node, list) else node
        for original, encoded in zip(messages, compact):
            self.assertEqual({k:v for k,v in original.items() if k != 'content'}, {k:v for k,v in encoded.items() if k != 'content'})
            if original['role'] != 'tool':
                self.assertEqual(original, encoded)
                continue
            node, end = json.JSONDecoder().raw_decode(encoded['content'])
            orig, orig_end = json.JSONDecoder().raw_decode(original['content'])
            self.assertEqual(decode(node), orig)
            tail = encoded['content'][end:].split('\nRepeated observation table ', 1)[0]
            for key, value in table.items():
                if isinstance(value, str):
                    tail = tail.replace('[' + marker + ':' + key + ']\n', value)
            self.assertEqual(tail, original['content'][orig_end:])

    def test_unique_history_is_not_shortened_by_truncating_evidence(self):
        _, e = case()
        messages = [{'role': 'tool', 'tool_call_id': 'one', 'content': 'unique facts ' * 30}]
        compact, info = e.compact_messages(messages)
        self.assertEqual(compact, messages)
        self.assertFalse(info)


class ReviewAndFinishContracts(unittest.TestCase):
    def review_case(self):
        ns, e = case(musts=['finish: matte'])
        e.rows['a'] = {'id': 'a', 'category': 'Widgets', 'price': 20, 'stock': True}
        e.full['a'] = {**e.rows['a'], 'attributes': {'finish': 'matte', 'material': 'metal'}}
        e.cart_lines, e.cart_touched = {'a': 1}, True
        e.judges = lambda limit=0: [ns['LIGHT']]
        return ns, e

    def test_rejected_review_survives_retry_until_evidence_changes(self):
        _, e = self.review_case()
        calls = []
        e.ask_json = lambda *a: calls.append(a) or {'ok': False, 'problems': ['Evidence conflict'], 'better': []}
        first = e.check_cart()
        self.assertTrue(first)
        self.assertEqual(e.check_cart(), first)
        self.assertEqual(len(calls), 1)
        self.assertTrue(e.terminal_problem())
        e.full['a']['attributes']['material'] = 'wood'
        e.ask_json = lambda *a: calls.append(a) or {'ok': True, 'problems': [], 'better': []}
        self.assertEqual(e.check_cart(), '')
        self.assertFalse(e.terminal_problem())
        self.assertEqual(len(calls), 2)

    def test_malformed_and_missing_reviews_never_become_approval(self):
        for response in (None, {}, {'ok': 'true'}, {'ok': 1}, {'ok': True, 'problems': ['Unresolved constraint']}, {'ok': True, 'better': ['a']}):
            _, e = self.review_case()
            e.ask_json = lambda *a: response
            self.assertTrue(e.check_cart(), response)
            self.assertIsNot(e.verdicts.get(e.check_key(['a'])), True)

    def test_fallback_cannot_skip_review_of_ambiguous_properties(self):
        _, e = self.review_case()
        e.full['a']['attributes']['widget finish'] = 'glossy'
        self.assertTrue(e.terminal_problem())
        e.ask_json = lambda *a: {'ok': True, 'problems': [], 'better': []}
        self.assertEqual(e.check_cart(), '')
        self.assertEqual(e.terminal_problem(), '')

    def test_interrupted_review_cannot_clear_on_retry(self):
        _, e = self.review_case()
        def interrupted(*args):
            raise RuntimeError('scripted interruption')
        e.ask_json = interrupted
        with self.assertRaises(RuntimeError):
            e.check_cart()
        self.assertTrue(e.check_cart())
        self.assertTrue(e.terminal_problem())

    def test_rejected_checkout_is_not_repeated_without_new_evidence(self):
        from types import SimpleNamespace
        ns, e = self.review_case()
        e.roles = {'cart': 'bag', 'buy': 'order'}
        calls = []
        def post(path, body):
            calls.append(body)
            return SimpleNamespace(data={'step': body['step'], 'observation': {'issue': 'Not accepted'}, 'finished': False}, error=None)
        ns['_env'] = SimpleNamespace(post_verbose=post)
        self.assertIsNotNone(e.act('order', {}))
        self.assertIsNone(e.act('order', {}))
        self.assertEqual(len(calls), 1)
        e.full['a']['attributes']['material'] = 'wood'
        self.assertIsNotNone(e.act('order', {}))
        self.assertEqual(len(calls), 2)

    def test_requirement_and_budget_changes_invalidate_approval(self):
        _, e = self.review_case()
        e.ask_json = lambda *a: {'ok': True}
        e.check_cart()
        old = e.check_key(['a'])
        e.items[0].musts = ['finish: glossy']
        self.assertNotEqual(old, e.check_key(['a']))
        old = e.check_key(['a'])
        e.budget = 10
        self.assertNotEqual(old, e.check_key(['a']))

    def test_empty_stale_cart_finishes_without_buying(self):
        ns, e = self.review_case()
        e.roles = {'cart': 'bag', 'buy': 'order', 'refuse': 'decline'}
        e.schemas = {n: {'type': 'object', 'properties': {}} for n in e.roles.values()}
        e.cart_lines = {}
        e.cart_view = {'lines': [{'id': 'a', 'quantity': 1}]}
        e.cart_at, e.cart_seen_at, e.step, e.max_steps = 34, 23, 34, 38
        e.cap_seen = len(e.full)
        e.gate = lambda stage: (True, '')
        calls = []
        def act(name, args):
            calls.append(name)
            e.step += 1
            e.finished = name == 'decline'
            return {'observation': {}, 'step': e.step, 'finished': e.finished}
        e.act = act
        e.finish()
        self.assertEqual(calls, ['decline'])
        self.assertTrue(e.finished)

    def test_sold_out_cart_gets_repair_chance_then_legal_decline(self):
        _, e = self.review_case()
        e.roles = {'cart': 'bag', 'buy': 'order', 'refuse': 'decline'}
        e.schemas = {n: {'type': 'object', 'properties': {}} for n in e.roles.values()}
        e.unavailable.add('a')
        e.gate = lambda stage: (True, '')
        calls = []
        def act(name, args):
            calls.append(name)
            e.finished = name == 'decline'
            return {'observation': {}, 'step': 1, 'finished': e.finished}
        e.act = act
        self.assertTrue(e.finish(again=True))
        self.assertFalse(calls)
        e.finish()
        self.assertEqual(calls, ['decline'])


class AtomicQuestionContracts(unittest.TestCase):
    def test_one_unsuccessful_question_does_not_trigger_a_property_list(self):
        _, e = case()
        e.msg_limit, e.msg_used = 3, 1
        e.lanes = {1: ['material', 'form', 'size']}
        e.lane_found, e.lane_spent, e.lane_used = {1: 0}, {1: 1}, {1: set()}
        self.assertEqual(e.next_ask(e.lanes, e.lane_found, e.lane_spent, e.lane_used), (1, ['material']))

    def test_ambiguous_list_answer_does_not_settle_unanswered_properties(self):
        _, e = case()
        e.lanes = {1: ['material', 'form', 'size']}
        e.settle(1, ['material', 'form', 'size'], False, [], False)
        self.assertNotIn('form', e.lane_used[1])
        self.assertNotIn('size', e.lane_used[1])
        self.assertFalse(e.items[0].free)

    def test_conservative_shopper_discloses_only_for_a_single_property(self):
        class AtomicWorld(v5.ScriptedV5World):
            def env(self, path, payload):
                was_preferred = self.preferred
                reply = super().env(path, payload)
                action = payload['action']
                question = action['args'].get('text', '')
                if action['name'] == self.names['ask'] and 'marker' in question and (',' in question or ' or ' in question):
                    self.preferred = was_preferred
                    self.last_answer = 'I have no preference.'
                    reply.data['observation'] = self.last_answer
                return reply
        world = AtomicWorld(load(), preferences=True, procedure=True)
        trace = world.run()
        self.assertTrue(world.finished and world.contract_satisfied, trace[-12:])
        questions = [a['args']['text'] for a in world.actions if a['name'] == world.names['ask']]
        self.assertEqual(len(questions), 4)
        self.assertFalse(any(',' in q or ' or ' in q for q in questions))
        self.assertFalse(world.unknown_prompts)


class SearchProgressContracts(unittest.TestCase):
    def search_case(self, documented=True):
        ns, e = case()
        e.roles = {'search': 'catalog_query'}
        props = {'query_words': {'type': 'string'}, 'department': {'type': 'string'},
                 'mode': {'type': 'string', 'enum': ['alpha', 'beta']}}
        schema = {'type': 'object', 'properties': props, 'required': ['mode']}
        e.schemas = {'catalog_query': schema}
        description = 'Use alpha for keyword search, or beta for filter search.' if documented else 'Search the catalog.'
        e.tools = [{'type': 'function', 'function': {'name': 'catalog_query', 'description': description, 'parameters': schema}}]
        return ns, e

    def test_mode_is_derived_from_advertised_contract_with_renamed_values(self):
        _, e = self.search_case()
        args = {'mode': 'beta', 'query_words': 'matte durable', 'department': 'widgets'}
        result = e.keyword_args('catalog_query', args)
        self.assertEqual(result, {**args, 'mode': 'alpha'})
        self.assertEqual(args['mode'], 'beta')

    def test_pure_filter_and_undocumented_modes_remain_unchanged(self):
        _, e = self.search_case()
        args = {'mode': 'beta', 'department': 'widgets'}
        self.assertEqual(e.keyword_args('catalog_query', args), args)
        _, e = self.search_case(False)
        args['query_words'] = 'matte durable'
        self.assertEqual(e.keyword_args('catalog_query', args), args)

    def test_repeated_ids_with_updated_facts_count_as_new_evidence(self):
        _, e = self.search_case()
        old = {'id': 'item-alpha', 'category': 'Widgets', 'price': 20, 'stock': True}
        e.step = 1
        e.search_progress('catalog_query', {'mode': 'alpha', 'query_words': 'first'}, {'products': [old]})
        e.step = 2
        e.search_progress('catalog_query', {'mode': 'alpha', 'query_words': 'second'}, {'products': [dict(reversed(list(old.items())))]})
        self.assertEqual(e.search_stalls, 1)
        self.assertTrue(e.queued)
        e.step = 3
        e.search_progress('catalog_query', {'mode': 'alpha', 'query_words': 'second'}, {'products': [{**old, 'stock': False}]})
        self.assertEqual(e.search_stalls, 0)
        self.assertFalse(e.trace[-1]['repeated'])

    def test_corrected_mode_reaches_actual_environment_payload(self):
        from types import SimpleNamespace
        ns, e = self.search_case()
        calls = []
        def post(path, body):
            calls.append(body)
            return SimpleNamespace(data={'step': body['step'], 'finished': False, 'observation': {'products': []}}, error=None)
        ns['_env'] = SimpleNamespace(post_verbose=post)
        e.act('catalog_query', {'mode': 'beta', 'query_words': 'matte'})
        self.assertEqual(calls[0]['action']['args']['mode'], 'alpha')
        self.assertEqual(calls[0]['step'], 1)


if __name__ == '__main__':
    unittest.main()
