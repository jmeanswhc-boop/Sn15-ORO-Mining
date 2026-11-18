"""Offline behavior regressions; scripted replies are not measured ORO scores."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from agents.tests import test_tosky_v4 as previous

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'agents/Tosky_v5.py'
original_load = previous.load


def load(path=SOURCE):
    return original_load(path)


def episode(request='Choose a suitable item.', name='sample product', musts=None):
    ns = load()
    e = previous.episode(ns)
    e.request = request
    e.items = [ns['Item'](name, '', 1, musts or [])]
    return ns, e


class RequestCoverageTests(unittest.TestCase):
    def audit(self, request, brief, answer):
        ns, e = episode(request)
        seen = []
        e.ask_json = lambda prompt, models, cap: seen.append((prompt, models, cap)) or answer
        e.audit_request(brief)
        return ns, e, seen

    def test_quoted_predicate_reaches_existing_checker(self):
        brief = {'items': [{'name': 'tile', 'musts': []}]}
        ns, e, seen = self.audit('I want a tile with a striped pattern.', brief,
            {'add': [{'item': 1, 'property': 'pattern', 'constraint': 'striped',
                      'quote': 'with a striped pattern', 'kind': 'property'}]})
        self.assertEqual(brief['items'][0]['musts'], ['pattern: striped'])
        e.items = [ns['Item']('tile', '', 1, brief['items'][0]['musts'])]
        e.full['candidate'] = {'attributes': {'pattern': 'plain', 'surface': 'rough'}}
        self.assertFalse(e.key_verdict('candidate', 'pattern: striped')[0])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1], [ns['LIGHT']])

    def test_product_purpose_is_preserved_without_a_property_table(self):
        brief = {'items': [{'name': 'lamp', 'musts': []}]}
        _, _, _ = self.audit("I need a lamp, and it's for outdoor work.", brief,
            {'add': [{'item': 1, 'property': 'intended use', 'constraint': 'outdoor work',
                      'quote': "it's for outdoor work", 'kind': 'purpose'}]})
        self.assertEqual(brief['items'][0]['musts'], ['intended use: outdoor work'])

    def test_background_story_cannot_be_added_as_product_purpose(self):
        brief = {'items': [{'name': 'lamp', 'musts': []}]}
        self.audit('My colleague is organising outdoor work. I need a lamp.', brief,
            {'add': [{'item': 1, 'property': 'intended use', 'constraint': 'outdoor work',
                      'quote': 'My colleague is organising outdoor work', 'kind': 'purpose'}]})
        self.assertEqual(brief['items'][0]['musts'], [])

    def test_type_is_a_predicate_even_when_present_in_product_name(self):
        brief = {'items': [{'name': 'lantern', 'musts': []}]}
        self.audit('I need a light and the type of light is a lantern.', brief,
            {'add': [{'item': 1, 'property': 'type of light', 'constraint': 'lantern',
                      'quote': 'the type of light is a lantern', 'kind': 'property'}]})
        self.assertEqual(brief['items'][0]['musts'], ['type of light: lantern'])

    def test_explicit_alternatives_keep_both_catalog_categories(self):
        brief = {'items': [{'name': 'stool', 'shelf': '', 'musts': ['finish: matte']}]}
        ns, e, _ = self.audit('I need a stool or bench with a matte finish.', brief,
            {'alternatives': [{'item': 1, 'names': ['stool', 'bench'], 'quote': 'a stool or bench'}]})
        self.assertEqual(brief['items'][0]['name'], 'stool or bench')
        e.items = [ns['Item'](brief['items'][0]['name'], '', 1, ['finish: matte'])]
        self.assertEqual(e.owner_of('Stools'), 1)
        self.assertEqual(e.owner_of('Benches'), 1)
        self.assertEqual(e.owner_of('Sofas'), 0)

    def test_property_alternatives_do_not_become_product_categories(self):
        brief = {'items': [{'name': 'stool', 'musts': ['color: red or blue']}]}
        self.audit('I need a stool in red or blue.', brief,
            {'alternatives': [{'item': 1, 'names': ['red', 'blue'], 'quote': 'red or blue'}]})
        self.assertEqual(brief['items'][0]['name'], 'stool')

    def test_fabricated_quote_value_or_property_cannot_change_requirements(self):
        for amendment in [
            {'quote': 'a plain pattern', 'property': 'pattern', 'constraint': 'plain'},
            {'quote': 'a striped pattern', 'property': 'pattern', 'constraint': 'plain'},
            {'quote': 'a striped pattern', 'property': 'material', 'constraint': 'striped'},
        ]:
            with self.subTest(amendment=amendment):
                brief = {'items': [{'name': 'tile', 'musts': []}]}
                self.audit('I want a tile with a striped pattern.', brief,
                           {'add': [{'item': 1, 'kind': 'property', **amendment}]})
                self.assertEqual(brief['items'][0]['musts'], [])

    def test_existing_constraints_and_later_relaxation_remain_effective(self):
        brief = {'items': [{'name': 'tile', 'musts': ['pattern: striped']}]}
        ns, e, _ = self.audit('I need a striped pattern, not plain.', brief,
            {'add': [{'item': 1, 'property': 'pattern', 'constraint': 'plain',
                      'quote': 'pattern, not plain', 'kind': 'property'}]})
        self.assertEqual(brief['items'][0]['musts'], ['pattern: striped'])
        e.items = [ns['Item']('tile', '', 1, brief['items'][0]['musts'])]
        e.take_heard({'drop': [{'item': 1, 'property': 'pattern'}]}, [])
        self.assertEqual(e.needs_now(), [])

    def test_unavailable_or_malformed_audit_preserves_original(self):
        for reply in (None, {}, {'add': [None, {'item': 90, 'quote': 'x'}]}):
            brief = {'items': [{'name': 'tile', 'musts': ['pattern: striped']}]}
            self.audit('Choose the tile.', brief, reply)
            self.assertEqual(brief['items'][0]['musts'], ['pattern: striped'])

    def test_negation_alternatives_and_bounds_cannot_be_stripped(self):
        for quote, value in [('pattern not plain', 'plain'), ('pattern red or blue', 'red'),
                             ('length at least 8', '8')]:
            with self.subTest(quote=quote):
                brief = {'items': [{'name': 'tile', 'musts': []}]}
                self.audit('I need a tile with ' + quote + '.', brief,
                    {'add': [{'item': 1, 'kind': 'property', 'property': quote.split()[0],
                              'constraint': value, 'quote': quote}]})
                self.assertEqual(brief['items'][0]['musts'], [])

    def test_quote_cannot_crop_off_an_immediately_preceding_negation(self):
        brief = {'items': [{'name': 'tile', 'musts': []}]}
        self.audit('I do not want a plain pattern on the tile.', brief,
            {'add': [{'item': 1, 'kind': 'property', 'property': 'pattern',
                      'constraint': 'plain', 'quote': 'a plain pattern'}]})
        self.assertEqual(brief['items'][0]['musts'], [])

    def test_multi_item_quote_must_identify_its_own_product(self):
        brief = {'items': [{'name': 'stool', 'musts': []}, {'name': 'lamp', 'musts': []}]}
        self.audit('I need a stool and a lamp. The lamp has a plain pattern.', brief,
            {'add': [{'item': 1, 'kind': 'property', 'property': 'pattern',
                      'constraint': 'plain', 'quote': 'The lamp has a plain pattern'}]})
        self.assertEqual(brief['items'][0]['musts'], [])


class InterviewCoverageTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.e = episode(name='sample product', musts=['finish: matte'])
        self.e.msg_limit, self.e.msg_used = 3, 1
        self.e.lanes = {1: ['finish', 'options', 'material', 'form', 'dimension', 'maker']}
        self.e.lane_found, self.e.lane_spent, self.e.lane_used = {1: 0}, {1: 1}, {1: set()}
        self.e.message_args = lambda text: {'text': text}

    def next(self):
        return self.e.next_ask(self.e.lanes, self.e.lane_found, self.e.lane_spent, self.e.lane_used)[1]

    def test_metadata_and_settled_properties_do_not_spend_questions(self):
        self.assertEqual(self.next(), ['material', 'form', 'dimension'])

    def test_first_question_is_single_and_preserves_planner_order(self):
        self.e.lane_spent[1], self.e.msg_used = 0, 0
        self.assertEqual(self.next(), ['material'])

    def test_catalog_diversity_does_not_override_meaningful_priority(self):
        self.e.item_ids = lambda n: ['a', 'b']
        self.e.full = {i: {'attributes': {'material': 'same', 'form': i, 'options': i}} for i in ['a', 'b']}
        self.assertEqual(self.e.focus_question(1, ['material', 'options', 'form']), 'material')

    def test_explicit_metadata_request_is_honored_alone(self):
        self.e.brief['ask_first'] = ['options']
        self.assertEqual(self.next(), ['options'])

    def test_required_single_questions_and_list_rejection_are_honored(self):
        self.e.brief['single_property_questions'] = True
        self.assertEqual(self.next(), ['material'])
        self.e.brief['single_property_questions'] = False
        self.e.settle(1, ['material', 'form'], False, [], True)
        self.assertFalse(self.e.listing)
        self.assertEqual(self.next(), ['material'])

    def test_message_reservation_and_character_limit_remain_effective(self):
        self.e.reserve = 1
        self.assertEqual(self.e.msgs_free(), 1)
        self.e.message_args = lambda text: {'text': text} if len(text) < 65 else None
        labels, text = self.e.fit_ask(1, self.next(), False)
        self.assertTrue(labels)
        self.assertLess(len(text), 65)
        self.e.msg_used = 2
        self.assertEqual(self.e.msgs_free(), 0)
        self.assertEqual(self.next(), [])

    def test_answered_or_explicitly_free_properties_are_not_reasked(self):
        self.e.items[0].told = ['material: woven']
        self.e.items[0].free = ['form']
        self.assertEqual(self.next(), ['dimension', 'maker'])

    def test_unsuccessful_short_list_advances_without_excluding_products(self):
        self.e.settle(1, ['material', 'form', 'dimension'], False, [], False)
        self.assertEqual(self.next(), ['maker'])
        self.assertEqual(self.e.items[0].musts, ['finish: matte'])
        self.assertEqual(self.e.items[0].told, [])


class PropertyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.e = episode(name='widget', musts=['finish: matte'])
        self.e.category_key = lambda: 'category'
        self.e.rows = {'a': {'id': 'a', 'category': 'Widgets'}}

    def record(self, attrs):
        self.e.full['a'] = {**self.e.rows['a'], 'attributes': attrs}

    def test_item_scoped_key_resolves_generic_conflict(self):
        self.record({'finish': 'glossy', 'widget finish': 'matte'})
        self.assertTrue(self.e.key_verdict('a', 'finish: matte')[0])
        self.assertIn('widget finish', self.e.shows_need('a', 'finish: matte'))

    def test_other_component_key_cannot_override_exact_key(self):
        self.record({'finish': 'glossy', 'handle finish': 'matte'})
        self.assertFalse(self.e.key_verdict('a', 'finish: matte')[0])
        self.assertEqual(self.e.shows_need('a', 'finish: matte'), '')

    def test_different_predicate_is_not_a_type_match(self):
        self.record({'widget type': 'basic', 'widget style': 'ornate'})
        self.assertFalse(self.e.key_verdict('a', 'type of widget: ornate')[0])

    def test_bounds_and_figures_follow_the_same_item_scope(self):
        self.record({'length': '9 m', 'widget length': '5 m'})
        self.assertTrue(self.e.bound_verdict('a', 'length', 'at most 5 m')[0])
        self.assertTrue(self.e.key_verdict('a', 'length: 5 m')[0])
        self.assertFalse(self.e.figure_conflict('a', 'length: 5 m'))

    def test_conflicting_scoped_evidence_stays_unresolved(self):
        self.e.rows['a']['category'] = 'Gadget Widgets'
        self.record({'widget finish': 'matte', 'gadget finish': 'glossy'})
        self.assertIsNone(self.e.key_verdict('a', 'finish: matte')[0])

    def test_explicit_exclusion_remains_enforced(self):
        self.record({'finish': 'glossy', 'widget finish': 'matte'})
        self.assertFalse(self.e.key_verdict('a', 'finish: not matte')[0])


class RecoveryCandidateTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.e = episode(name='widget', musts=['finish: matte'])
        self.e.brief = {'pick': 'cheapest'}
        self.e.category_key = lambda: 'category'
        self.e.id_key, self.e.price_key, self.e.stock_key = 'id', 'cost', 'available'
        self.e.rows = {i: {'id': i, 'category': 'Widgets', 'cost': price, 'available': True}
                       for i, price in [('lost', 1), ('best', 2)] + [(f'later-{n}', n + 10) for n in range(40)]}
        self.e.full = {i: {**r, 'attributes': {'finish': 'matte', 'count': 'one'}} for i, r in self.e.rows.items()}
        self.e.seen_order = list(self.e.rows)
        self.e.unavailable.add('lost')
        self.prompts = []
        self.e.ask_json = lambda prompt, *args: self.prompts.append(prompt) or {'fits': [{'id': 'later-39'}]}

    def test_old_best_opened_candidate_is_present_and_remains_first(self):
        result = self.e.replacements(['lost'])
        self.assertIn('"best"', self.prompts[0])
        self.assertEqual(result[0], 'best')
        self.assertLessEqual(len(self.e.recovery_pool(['lost'])), 25)

    def test_stale_best_and_violating_candidate_are_not_rescued(self):
        self.e.unavailable.add('best')
        self.e.full['later-0']['attributes']['finish'] = 'glossy'
        result = self.e.replacements(['lost'])
        self.assertNotIn('best', result)
        self.assertNotIn('later-0', result)

    def test_current_relaxation_changes_eligibility_without_stale_cache(self):
        self.e.full['best']['attributes']['finish'] = 'glossy'
        self.assertNotIn('best', self.e.replacements(['lost']))
        self.e.take_heard({'drop': [{'item': 1, 'property': 'finish'}]}, [])
        self.assertEqual(self.e.replacements(['lost'])[0], 'best')

    def test_known_alternative_category_is_kept_but_unrelated_kind_is_not(self):
        self.e.items[0].name = 'widget or gadget'
        self.e.rows['best']['category'] = self.e.full['best']['category'] = 'Gadgets'
        self.e.rows['later-0']['category'] = self.e.full['later-0']['category'] = 'Chairs'
        self.assertEqual(self.e.replacements(['lost'])[0], 'best')
        self.assertNotIn('later-0', self.e.recovery_pool(['lost']))

    def test_joint_compatibility_is_not_inferred_from_individual_predicates(self):
        self.e.bound = [(1, 2)]
        self.assertEqual(self.e.replacements(['lost']), ['later-39'])


class ScriptedV5World(previous.ScriptedWorld):
    def model(self, path, body):
        content = (body.get('messages') or [{}])[0].get('content', '')
        prompt = content if isinstance(content, str) else content[0].get('text', '')
        if not body.get('tools') and prompt.startswith(self.ns.get('REQUEST_AUDIT', '\0')[:60]):
            self.model_calls.append(body)
            return previous.SimpleNamespace(data={'choices': [{'message': {'content': '{"add":[],"alternatives":[]}'}}],
                                                  'usage': {'cost': 0.00001}}, error=None)
        if not body.get('tools') and prompt.startswith(self.ns['AUDIT'][:60]) and self.decline:
            self.model_calls.append(body)
            return previous.SimpleNamespace(data={'choices': [{'message': {'content': '{"fits":{},"open":[]}'}}],
                                                  'usage': {'cost': 0.00001}}, error=None)
        return super().model(path, body)


class RecoveryWindowTests(unittest.TestCase):
    def test_first_order_reserves_retry_and_two_cart_mutations(self):
        ns, e = episode()
        self.assertEqual(e.recovery_room(), 4)
        e.process = [{'when': 'order', 'step': 'Explain before ordering', 'messages': 1}]
        e.msg_limit = 1
        self.assertEqual(e.recovery_room(), 5)
        e.buys = 1
        self.assertEqual(e.recovery_room(), e.tail())

    def test_late_cart_fill_leaves_room_for_actual_stockout_recovery(self):
        class SearchingWorld(ScriptedV5World):
            def problem(self):
                p = super().problem()
                p['environment']['policy_view']['max_steps'] = 16
                return p

            def model(self, path, body):
                reply = super().model(path, body)
                if body.get('tools') and not self.cart and not self.changed:
                    fn = reply.data['choices'][0]['message']['tool_calls'][0]['function']
                    fn.update(name=self.names['search'], arguments='{"query":"sample products"}')
                return reply

        world = SearchingWorld(load(), stockout=True)
        trace = world.run()
        self.assertFalse(world.unknown_prompts)
        self.assertTrue(world.finished and world.contract_satisfied, trace[-12:])
        self.assertLessEqual(len(world.actions), 16)
        first_buy = next(i for i, a in enumerate(world.actions, 1) if a['name'] == world.names['buy'])
        self.assertGreaterEqual(16 - first_buy, 3)
        self.assertEqual(world.cart, {'sample-B02': 1})


class ProcedureReuseTests(unittest.TestCase):
    def setUp(self):
        self.ns = load()
        self.e = previous.procedure_episode(self.ns)
        del self.e.ledger
        self.e.done_log, self.e.done_steps = ['An observed explanation was delivered.'], [0]
        self.calls = []
        self.e.ask_json = lambda *args: self.calls.append(args) or {'steps': [{'duty': 1, 'done': True, 'lines': [1]}]}

    def test_proven_general_explanation_survives_inventory_and_cart_change(self):
        self.assertTrue(self.e.gate('order')[0])
        self.e.version += 1
        self.e.cart_lines = {'another-item': 1}
        self.e.cart_at = 9
        self.assertTrue(self.e.gate('order')[0])
        self.assertEqual(len(self.calls), 1)

    def test_changed_requirements_budget_or_duty_invalidate_general_proof(self):
        self.assertTrue(self.e.gate('order')[0])
        self.e.items[0].told.append('finish: matte')
        self.assertTrue(self.e.gate_due('order'))
        self.assertTrue(self.e.gate('order')[0])
        self.e.budget = 7
        self.assertTrue(self.e.gate_due('order'))
        self.assertTrue(self.e.gate('order')[0])
        self.e.process[0]['step'] = 'Give a different explanation'
        self.assertTrue(self.e.gate_due('order'))

    def test_cart_dependent_approval_requires_new_evidence(self):
        self.e.process[0]['cart_dependent'] = True
        self.e.done_steps = [2]
        self.assertTrue(self.e.gate('order')[0])
        self.e.cart_lines = {'another-item': 1}
        self.e.cart_at = 9
        self.e.version += 1
        self.assertFalse(self.e.gate('order')[0])


class IntegratedPreservationTests(unittest.TestCase):
    def test_previous_successful_scripted_flows_remain_successful(self):
        cases = [{}, {'procedure': True}, {'stockout': True},
                 {'decline': True, 'prefix': 'changed_'}, {'preferences': True, 'procedure': True},
                 {'procedure': True, 'stockout': True}]
        for case in cases:
            with self.subTest(case=case):
                before = ScriptedV5World(load(ROOT / 'agents/Tosky_v4.py'), **case)
                after = ScriptedV5World(load(), **case)
                before_trace, after_trace = before.run(), after.run()
                self.assertTrue(before.finished and before.contract_satisfied, before_trace[-8:])
                self.assertTrue(after.finished and after.contract_satisfied, after_trace[-8:])
                self.assertFalse(after.unknown_prompts)
                self.assertFalse([r for r in after_trace if r.get('kind') in ('crash', 'soft_crash')])
                self.assertLessEqual(len(after.actions), len(before.actions))
                self.assertLessEqual(sum(a['name'] == after.names['ask'] for a in after.actions),
                                     sum(a['name'] == before.names['ask'] for a in before.actions))


class PreservedMechanicsTests(unittest.TestCase):
    def test_existing_budget_procedure_change_and_reference_contracts(self):
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
            for cls in [previous.BudgetTests, previous.ProcedureTests, previous.ChangeTests, previous.EvidenceTests])
        with patch.object(previous, 'load', side_effect=lambda path=None: original_load(path or SOURCE)):
            result = unittest.TestResult()
            suite.run(result)
        self.assertEqual(result.errors + result.failures, [])
        self.assertEqual(result.testsRun, 36)


if __name__ == '__main__':
    unittest.main()
