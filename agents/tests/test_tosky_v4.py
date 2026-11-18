"""Offline behavior checks. Scripted model replies are not benchmark scores."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'agents/Tosky_v4.py'
BASE = ROOT / 'champions/PIRATE_v1_06af6030-67a0-754e-8000-beacfd741adb/agent.py'


class OfflineClient:
    def __init__(self, **kwargs):
        self.options = kwargs

    def post_verbose(self, *args, **kwargs):
        raise AssertionError('Unscripted transport is forbidden in offline checks')


def load(path=SOURCE):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not
                 (isinstance(n, ast.ImportFrom) and n.module == 'src.agent.proxy_client')]
    ns = {'__name__': 'offline_tosky', 'ProxyClient': OfflineClient}
    exec(compile(tree, str(path), 'exec'), ns)
    ns['time'] = SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _: None)
    return ns


def episode(ns):
    return ns['Cairn']({'environment': {'binding': {'opaque': 'binding-value'},
                         'policy_view': {'query': 'Choose a suitable item.', 'max_steps': 40,
                                         'tools': [], 'finished': False}}})


def transport(ns, replies):
    sent = []
    queue = iter(replies)

    def post(path, body):
        sent.append((path, body))
        value = next(queue)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(data=value[0], error=value[1])

    ns['_llm_within'] = lambda _: SimpleNamespace(post_verbose=post)
    return sent


def success(cost=0.00001, content='ok'):
    return ({'choices': [{'message': {'content': content}}],
             'usage': {} if cost is None else {'cost': cost}}, None)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.ns = load()
        self.e = episode(self.ns)

    def test_ordinary_request_keeps_model_prompt_and_output_limit(self):
        sent = transport(self.ns, [success()])
        messages = [{'role': 'user', 'content': 'Read these observations.'}]
        result = self.e.think(messages, [self.ns['LIGHT']], 1000)
        self.assertEqual(result['content'], 'ok')
        self.assertEqual(sent[0][1], {'model': self.ns['LIGHT'], 'messages': messages,
                                     'max_tokens': 1000, 'temperature': 0})
        self.assertEqual(self.e.calls, 1)
        self.assertAlmostEqual(self.e.cost, 0.00001)

    def test_402_stops_after_one_physical_request_including_fallbacks(self):
        sent = transport(self.ns, [(None, {'status': 402})])
        self.assertIsNone(self.e.think([], [self.ns['LIGHT'], self.ns['SPARE']], 1000))
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.e.calls, 1)
        self.assertTrue(self.e.no_credit)
        self.assertIsNone(self.e.think([], [self.ns['LIGHT']], 1000))
        self.assertEqual(len(sent), 1)

    def test_retry_costs_and_attempts_are_not_free(self):
        sent = transport(self.ns, [(None, {'status': 503}), (None, {'status': 429}), success()])
        self.assertIsNotNone(self.e.think([], [self.ns['LIGHT']], 1000))
        self.assertEqual((len(sent), self.e.calls), (3, 3))
        self.assertGreater(self.e.cost_uncertain, 0)

    def test_missing_invalid_and_nan_cost_keep_reservation(self):
        for cost in (None, True, -1, float('nan'), float('inf')):
            with self.subTest(cost=cost):
                e = episode(self.ns)
                transport(self.ns, [success(cost)])
                e.call_model(self.ns['LIGHT'], [], 1000, None)
                self.assertGreater(e.cost_uncertain, 0)
                self.assertEqual(e.cost_actual, 0)

    def test_uncertain_transport_stays_charged(self):
        transport(self.ns, [TimeoutError()])
        self.e.call_model(self.ns['LIGHT'], [], 1000, None)
        self.assertGreater(self.e.cost_uncertain, 0)
        self.assertEqual(self.e.calls, 1)

    def test_breach_stops_later_requests(self):
        sent = transport(self.ns, [success(0.1)])
        self.e.call_model(self.ns['LIGHT'], [], 1000, None)
        self.assertTrue(self.e.budget_breached)
        self.assertIsNone(self.e.think([], [self.ns['LIGHT']], 1000))
        self.assertEqual(len(sent), 1)

    def test_unpriced_model_and_exhausted_budget_never_send(self):
        sent = transport(self.ns, [])
        self.assertEqual(self.e.call_model('unpriced/model', [], 1000, None)[1], 'budget_denied')
        self.e.cost = self.ns['EPISODE_USD']
        self.assertIsNone(self.e.think([], [self.ns['LIGHT']], 1000))
        self.assertTrue(self.e.budget_denied)
        self.assertEqual(sent, [])

    def test_physical_call_limit_includes_retries(self):
        self.e.calls = self.ns['ALL_CALLS'] - 1
        sent = transport(self.ns, [(None, {'status': 503})])
        self.e.think([], [self.ns['LIGHT'], self.ns['SPARE']], 1000)
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.e.calls, self.ns['ALL_CALLS'])

    def test_long_context_tier_and_output_cap_are_budgeted(self):
        self.e.cost = 0.040
        body = {'model': self.ns['LIGHT'], 'messages': [{'role': 'user', 'content': 'x' * 100000}], 'max_tokens': 1000}
        self.assertIsNone(self.e.quote_inference(body))
        body['messages'][0]['content'] = 'x' * 1000
        body['model'] = self.ns['HEAVY']
        quote = self.e.quote_inference(body)
        self.assertIsNotNone(quote)
        self.assertLess(quote[1], 1000)
        self.assertLessEqual(quote[0] + self.e.cost, self.ns['EPISODE_USD'] + 1e-12)

    def test_unknown_billing_stress_90_invocations(self):
        total = 0.0
        for _ in range(90):
            e = episode(self.ns)
            transport(self.ns, [success(None)] * self.ns['ALL_CALLS'])
            for _ in range(self.ns['ALL_CALLS'] + 1):
                e.call_model(self.ns['LIGHT'], [], 3000, None)
            self.assertLessEqual(e.cost, self.ns['EPISODE_USD'] + 1e-12)
            total += e.cost
        self.assertLessEqual(total, 4.5 + 1e-9)


class PreservationTests(unittest.TestCase):
    def test_original_sources_unchanged(self):
        hashes = json.loads((ROOT/'logs/research/tosky-v4-build-2026-11-14/original-hashes.json').read_text())
        for filename, digest in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT/filename).read_bytes()).hexdigest(), digest)

    def test_inference_clients_do_not_retry_internally(self):
        ns = load()
        self.assertEqual(ns['_llm'].options['max_retries'], 1)
        for seconds in (10, 30, 100):
            self.assertEqual(ns['_llm_within'](seconds).options['max_retries'], 1)


def procedure_episode(ns):
    e = episode(ns)
    e.items = [ns['Item']('synthetic item', 'category', 1, [])]
    e.roles = {'buy': 'purchase', 'refuse': 'decline'}
    e.schemas = {name: {'type': 'object', 'properties': {}} for name in e.roles.values()}
    e.cart_lines = {'synthetic-item': 1}
    e.cart_at = 1
    e.step = 3
    e.process = [{'step': 'Inform the shopper before committing', 'when': 'order',
                  'messages': 1, 'cart_dependent': False}]
    e.ledger = lambda: 'A synthetic request with one procedure.'
    e.cart_text = lambda: 'one synthetic item'
    return e


class ProcedureTests(unittest.TestCase):
    def setUp(self):
        self.ns = load()
        self.e = procedure_episode(self.ns)

    def test_unavailable_review_does_not_grant_permission(self):
        self.e.ask_json = lambda *args: None
        self.assertFalse(self.e.gate('order')[0])

    def test_empty_or_incomplete_review_does_not_grant_permission(self):
        for answer in ({}, {'steps': []}, {'steps': [{'duty': 1, 'done': False, 'lines': []}]}):
            with self.subTest(answer=answer):
                e = procedure_episode(self.ns)
                e.ask_json = lambda *args: answer
                self.assertFalse(e.gate('order')[0])

    def test_complete_observed_duty_allows_order_and_reuses_decision(self):
        self.e.done_log = ['The shopper was informed.']
        self.e.done_steps = [0]
        calls = []
        def answer(*args):
            calls.append(args)
            return {'steps': [{'duty': 1, 'step': self.e.process[0]['step'], 'done': True, 'lines': [1]}], 'call': None}
        self.e.ask_json = answer
        self.assertTrue(self.e.gate('order')[0])
        self.assertTrue(self.e.gate('order')[0])
        self.assertEqual(len(calls), 1)

    def test_old_cart_approval_is_not_current_cart_approval(self):
        self.e.process[0]['cart_dependent'] = True
        self.e.done_log = ['An earlier cart was approved.']
        self.e.done_steps = [0]
        self.e.ask_json = lambda *args: {'steps': [{'duty': 1, 'step': self.e.process[0]['step'], 'done': True, 'lines': [1]}]}
        self.assertFalse(self.e.gate('order')[0])

    def test_invalid_record_line_is_not_completion(self):
        self.e.done_log = ['A failed action.']
        self.e.done_steps = [-1]
        self.e.ask_json = lambda *args: {'steps': [{'duty': 1, 'done': True, 'lines': [1, 9]}]}
        self.assertFalse(self.e.gate('order')[0])

    def test_unchanged_hold_does_not_restart_review_after_unrelated_reads(self):
        calls = []
        self.e.ask_json = lambda *args: calls.append(args)
        for step in range(3, 12):
            self.e.step = step
            self.e.full['unrelated'] = {'read_step': step}
            self.assertFalse(self.e.gate('order')[0])
        self.assertLessEqual(len(calls), 2)

    def test_no_procedure_needs_no_model(self):
        self.e.process = []
        self.e.ask_json = lambda *args: self.fail('No review is required')
        self.assertEqual(self.e.gate('order'), (True, ''))

    def test_both_fallback_order_paths_honor_negative_gate(self):
        for name in ('direct_order', 'finish'):
            with self.subTest(name=name):
                e = procedure_episode(self.ns)
                e.roles.pop('refuse')
                e.cart_view = {}
                e.cart_seen_at = e.step
                e.category_key = lambda: 'category'
                e.cart_ids = lambda: ['synthetic-item']
                e.over_lines = e.short_items = e.line_faults = lambda: []
                e.unfit = lambda _: False
                e.gate = lambda _: (False, 'An actual duty is unresolved.')
                sent = []
                e.act = lambda *args: sent.append(args)
                getattr(e, name)()
                self.assertEqual(sent, [])

    def test_lowest_terminal_dispatch_blocks_missing_permission(self):
        sent = []
        self.ns['_env'] = SimpleNamespace(post_verbose=lambda *args: sent.append(args))
        self.assertIsNone(self.e.act('purchase', {}))
        self.assertEqual(sent, [])


    def test_informing_then_reviewing_completes_without_asking_for_approval(self):
        self.e.roles['ask'] = 'contact'
        self.e.schemas['contact'] = {'type': 'object', 'properties': {'text': {'type': 'string'}}}
        self.e.tools = [{'type': 'function', 'function': {'name': 'contact', 'parameters': self.e.schemas['contact']}}]
        self.e.msg_limit = 1
        self.e.message_args = lambda text: {'text': text}
        sent = []
        def act(name, args):
            sent.append((name, args))
            self.e.step += 1
            return {'step': self.e.step, 'observation': 'Thank you.', 'finished': False}
        self.e.act = act
        replies = iter([
            {'steps': [{'duty': 1, 'done': False, 'lines': []}], 'call': {'tool': 'contact', 'args': {'text': 'Check the supplied details before committing.'}}},
            {'steps': [{'duty': 1, 'done': True, 'lines': [1]}], 'call': None},
        ])
        self.e.ask_json = lambda *args: next(replies)
        self.assertTrue(self.e.gate('order')[0])
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.e.msg_used, 1)
        self.assertFalse(self.e.in_procedure)

    def test_combined_terminal_tool_discriminates_arguments(self):
        self.e.roles = {'buy': 'finish', 'refuse': 'finish'}
        self.e.brief = {'buy_args': {'mode': 'buy'}, 'refuse_args': {'mode': 'decline'}}
        self.assertTrue(self.e.is_buy('finish', {'mode': 'buy'}))
        self.assertFalse(self.e.is_buy('finish', {'mode': 'decline'}))
        sent = []
        self.ns['_env'] = SimpleNamespace(post_verbose=lambda *args: sent.append(args))
        self.assertIsNone(self.e.act('finish', {'mode': 'buy'}))
        self.assertEqual(sent, [])


class ChangeTests(unittest.TestCase):
    def setUp(self):
        self.ns = load()
        self.e = procedure_episode(self.ns)
        self.e.process = []
        self.e.rows = {'synthetic-item': {'sku': 'synthetic-item', 'cost': 4, 'available': True}}
        self.e.stock_key = 'available'
        self.e.price_key = 'cost'
        self.e.id_key = 'sku'
        self.e.attempted_cart = {'synthetic-item': 1}
        self.e.attempted_step = self.e.step
        self.e.ask_json = lambda *args: {'sold_out': True, 'gone': [], 'kind': 'stock'}

    def test_unnamed_single_line_stockout_invalidates_attempted_line(self):
        self.e.take_note(['The selected item is no longer available.'])
        self.assertTrue(self.e.sold_out('synthetic-item'))

    def test_ambiguous_stockout_marks_scope_without_declaring_every_line_gone(self):
        self.e.rows['other-item'] = {'sku': 'other-item', 'available': True}
        self.e.attempted_cart['other-item'] = 1
        self.e.take_note(['One of the selected items became unavailable.'])
        self.assertFalse(self.e.sold_out('synthetic-item'))
        self.assertFalse(self.e.sold_out('other-item'))
        self.assertEqual(set(getattr(self.e, 'availability_pending', {})), {'synthetic-item', 'other-item'})

    def test_unrelated_boolean_change_does_not_mean_sold_out(self):
        self.e.rows['synthetic-item']['feature_flag'] = True
        self.e.absorb({'sku': 'synthetic-item', 'cost': 4, 'feature_flag': False})
        self.assertFalse(self.e.sold_out('synthetic-item'))

    def test_price_and_quantity_change_invalidate_review(self):
        key = self.e.check_key(['synthetic-item'])
        self.e.rows['synthetic-item']['cost'] = 8
        self.assertNotEqual(key, self.e.check_key(['synthetic-item']))
        key = self.e.check_key(['synthetic-item'])
        self.e.cart_lines['synthetic-item'] = 2
        self.assertNotEqual(key, self.e.check_key(['synthetic-item']))

    def test_unrelated_product_read_preserves_review(self):
        key = self.e.check_key(['synthetic-item'])
        self.e.rows['unrelated'] = {'cost': 999}
        self.assertEqual(key, self.e.check_key(['synthetic-item']))

    def test_unavailable_item_cannot_bypass_last_step_guard(self):
        self.e.unavailable.add('synthetic-item')
        self.e.step = self.e.max_steps - 1
        sent = []
        self.ns['_env'] = SimpleNamespace(post_verbose=lambda *args: sent.append(args))
        self.assertIsNone(self.e.act('purchase', {}))
        self.assertEqual(sent, [])

    def test_old_attempt_does_not_capture_an_unrelated_later_notice(self):
        self.e.step += 1
        self.e.take_note(['A different selection has become unavailable.'])
        self.assertFalse(self.e.sold_out('synthetic-item'))

    def test_only_a_fresh_offer_resolves_ambiguous_availability(self):
        self.e.roles['offer'] = 'offer'
        self.e.availability_pending = {'synthetic-item': self.e.step}
        record = {'sku': 'synthetic-item', 'cost': 4, 'available': True}
        self.e.refresh_availability('offer', record)
        self.assertIn('synthetic-item', self.e.availability_pending)
        self.e.step += 1
        self.e.refresh_availability('attributes', record)
        self.assertIn('synthetic-item', self.e.availability_pending)
        self.e.refresh_availability('offer', record)
        self.assertNotIn('synthetic-item', self.e.availability_pending)
        self.assertFalse(self.e.sold_out('synthetic-item'))

    def test_fresh_negative_offer_resolves_to_unavailable(self):
        self.e.roles['offer'] = 'offer'
        self.e.availability_pending = {'synthetic-item': self.e.step - 1}
        self.e.refresh_availability('offer', {'sku': 'synthetic-item', 'cost': 4, 'available': False})
        self.assertNotIn('synthetic-item', self.e.availability_pending)
        self.assertTrue(self.e.sold_out('synthetic-item'))

    def test_informational_note_preserves_review_and_retry_state(self):
        self.e.ask_json = lambda *args: {'kind': 'change', 'summary': 'The store provided background information.'}
        key = self.e.check_key(['synthetic-item'])
        self.e.take_note(['Background information without a changed requirement.'])
        self.assertEqual(key, self.e.check_key(['synthetic-item']))

    def test_replacements_use_observed_prices_instead_of_invented_quotes(self):
        self.e.brief['pick'] = 'cheapest'
        self.e.rows = {i: {'id': i, 'category': 'sample', 'cost': price}
                       for i, price in [('synthetic-item', 9), ('choice-A', 2), ('choice-B', 4)]}
        self.e.full = dict(self.e.rows)
        self.e.seen_order = list(self.e.rows)
        self.e.category_key = lambda: 'category'
        self.e.breaks_need = self.e.unfit = lambda *args: False
        self.e.ask_json = lambda *args: {'fits': [{'id': 'choice-B', 'price': 1}, {'id': 'choice-A', 'price': 99}]}
        self.assertEqual(self.e.replacements(['synthetic-item']), ['choice-A', 'choice-B'])



class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.ns = load()
        self.e = episode(self.ns)
        self.e.not_maker_keys = lambda: set()
        self.e.full['reference'] = {'attributes': {'measurement': '10 V'}}

    def test_different_measurements_do_not_override_a_negative_review(self):
        self.e.ask_json = lambda *args: {'match': False, 'missing': ['1000 mAh']}
        self.assertTrue(self.e.link_mismatch('reference', 'Required measurement 1000 mAh.'))

    def test_semantically_confirmed_conversion_needs_no_additional_judge(self):
        calls = []
        self.e.ask_json = lambda *args: calls.append(args) or {'match': True, 'missing': []}
        self.assertFalse(self.e.link_mismatch('reference', 'A supplied description with a converted value.'))
        self.assertEqual(len(calls), 1)

    def test_changed_reference_is_rechecked_but_unchanged_one_is_cached(self):
        calls = []
        self.e.ask_json = lambda *args: calls.append(args) or {'match': True}
        self.e.link_mismatch('reference', 'A description')
        self.e.link_mismatch('reference', 'A description')
        self.assertEqual(len(calls), 1)
        self.e.full['reference']['attributes']['measurement'] = '20 V'
        self.e.link_mismatch('reference', 'A description')
        self.assertEqual(len(calls), 2)

    def test_same_unit_numeric_bounds_preserve_edges_and_ambiguity(self):
        self.e.full['reference'] = {'attributes': {'measurement': '5 m', 'other property': 'observed'}}
        self.assertTrue(self.e.bound_verdict('reference', 'measurement', 'at most 5 m')[0])
        self.assertFalse(self.e.bound_verdict('reference', 'measurement', 'at most 4 m')[0])
        self.assertFalse(self.e.bound_verdict('reference', 'measurement', 'less than 5 m')[0])
        self.e.full['reference']['attributes']['measurement'] = '3 m, 6 m'
        self.assertIsNone(self.e.bound_verdict('reference', 'measurement', 'at most 5 m')[0])


class InterviewTests(unittest.TestCase):
    def setUp(self):
        self.ns = load()
        self.e = episode(self.ns)
        self.e.items = [self.ns['Item']('synthetic item', 'category', 1, [])]
        self.e.msg_limit = 3
        self.e.item_ids = lambda number: ['a', 'b', 'c']
        self.e.full = {i: {'attributes': {'uniform': 'same', 'varied': str(n)}} for n, i in enumerate(['a', 'b', 'c'])}
        self.e.lanes = {1: ['uniform', 'varied']}
        self.e.lane_found = {1: 0}
        self.e.lane_spent = {1: 1}
        self.e.lane_used = {1: set()}
        self.e.message_args = lambda text: {'text': text}

    def test_one_clear_property_per_item_question(self):
        labels, text = self.e.fit_ask(1, ['uniform', 'varied'], False)
        self.assertEqual(labels, ['uniform'])
        self.assertNotIn('varied', text)

    def test_after_unproductive_question_use_observed_discrimination(self):
        number, labels = self.e.next_ask(self.e.lanes, self.e.lane_found, self.e.lane_spent, self.e.lane_used)
        self.assertEqual((number, labels), (1, ['varied']))

    def test_first_question_keeps_original_planner_priority(self):
        self.e.lane_spent[1] = 0
        number, labels = self.e.next_ask(self.e.lanes, self.e.lane_found, self.e.lane_spent, self.e.lane_used)
        self.assertEqual((number, labels), (1, ['uniform']))

    def test_explicit_ask_first_is_not_replaced_by_discrimination(self):
        self.e.brief['ask_first'] = ['uniform']
        self.assertEqual(self.e.next_ask(self.e.lanes, self.e.lane_found, self.e.lane_spent, self.e.lane_used)[1], ['uniform'])

    def test_reply_can_explicitly_name_an_unseen_property(self):
        self.e.quiet_reply = lambda _: False
        self.e.plain_value = lambda *args: ''
        self.e.ask_json = lambda *args: {'needs': [{'item': 1, 'need': 'new field: specific', 'property_quote': 'new field'}]}
        result = self.e.hear([(1, 'Which uniform value?', 'For new field, I require specific.')], (self.ns['LIGHT'],), 1, ('uniform',))
        self.assertEqual(result['needs'][0]['need'], 'new field: specific')

    def test_fabricated_property_quote_is_not_accepted(self):
        self.e.quiet_reply = lambda _: False
        self.e.plain_value = lambda *args: ''
        self.e.ask_json = lambda *args: {'needs': [{'item': 1, 'need': 'invented field: specific', 'property_quote': 'invented field'}]}
        result = self.e.hear([(1, 'Which uniform value?', 'I have no preference.')], (self.ns['LIGHT'],), 1, ('uniform',))
        self.assertEqual(result['needs'], [])


class ScriptedWorld:
    """A protocol fixture with a scripted model, not an ORO world or grader."""
    def __init__(self, ns, procedure=False, stockout=False, decline=False, prefix='fixture_', preferences=False):
        self.ns, self.procedure, self.stockout, self.decline = ns, procedure, stockout, decline
        self.preferences, self.preferred, self.last_answer = preferences, False, ''
        self.names = {r: prefix + r for r in ('search', 'open', 'cart', 'cart_view', 'ask', 'buy', 'refuse')}
        self.cart, self.actions, self.model_calls, self.unknown_prompts = {}, [], [], []
        self.talk, self.changed, self.finished = False, False, False
        self.contract_satisfied = False
        self.products = {i: {'id': i, 'category': 'sample products', 'title': i,
                             'price': price, 'stock': True,
                             'attributes': {'surface': 'smooth', 'marker': 'plain'}}
                         for i, price in [('sample-A01', 1), ('sample-B02', 2)]}
        self.binding = {'session_id': 'synthetic-session', 'opaque_extension': {'keep': 'verbatim'}}
        if preferences:
            for index, record in enumerate(self.products.values()):
                record['attributes'].update({'marker': 'first' if index == 0 else 'second', 'uniform': 'shared', 'unvarying': 'shared'})
        self.tools = []
        specs = {
            'search': ('Search sample products.', {'query': {'type': 'string'}}),
            'open': ('Read one product including attributes, price and stock.', {'id': {'type': 'string'}}),
            'cart': ('Set the quantity of one product in the cart; zero removes it.', {'id': {'type': 'string'}, 'quantity': {'type': 'integer'}}),
            'cart_view': ('Show the current cart.', {}),
            'ask': ('Send a message to the shopper.', {'text': {'type': 'string'}}),
            'buy': ('Purchase the cart and finish.', {}),
            'refuse': ('Finish without a purchase.', {}),
        }
        for role, (description, properties) in specs.items():
            self.tools.append({'type': 'function', 'function': {'name': self.names[role], 'description': description,
                               'parameters': {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}}})

    def problem(self):
        return {'environment': {'binding': self.binding, 'policy_view': {
            'query': 'Choose one sample product with surface ' + ('absent' if self.decline else 'smooth') + '. ' + ('Explain the checks before purchasing.' if self.procedure else ''),
            'max_steps': 35, 'tools': self.tools,
            'opening': {'products': list(self.products.values()), 'count': 2}, 'finished': False}}}

    def env(self, path, payload):
        assert path == '/environment/call'
        assert all(payload[k] == v for k, v in self.binding.items())
        assert payload['step'] == len(self.actions) + 1
        action = payload['action']
        self.actions.append(action)
        role = next(r for r, n in self.names.items() if n == action['name'])
        args = action['args']
        observation = {}
        if role == 'search':
            observation = {'products': list(self.products.values()), 'count': 2}
        elif role == 'open':
            observation = self.products[args['id']]
        elif role == 'cart':
            if args['quantity'] > 0:
                self.cart[args['id']] = args['quantity']
            else:
                self.cart.pop(args['id'], None)
            observation = {'id': args['id'], 'quantity': self.cart.get(args['id'], 0)}
        elif role == 'cart_view':
            observation = {'lines': [{'id': i, 'quantity': q} for i, q in self.cart.items()]}
        elif role == 'ask':
            if args['text'].startswith('Check '):
                self.talk = True
                observation = 'Thank you for explaining the checks.'
            elif self.preferences and 'marker' in args['text']:
                self.preferred = True
                observation = 'For marker, I require second.'
            else:
                observation = 'I have no preference.'
            self.last_answer = observation
        elif role == 'buy':
            if self.stockout and not self.changed:
                self.changed = True
                self.products['sample-A01']['stock'] = False
                observation = {'note': 'The selected item is sold out. Please choose a replacement.'}
            else:
                self.contract_satisfied = bool(not self.decline and self.cart and
                    all(self.products[i]['stock'] for i in self.cart) and
                    (not self.procedure or self.talk) and
                    (not self.preferences or (self.preferred and set(self.cart) == {'sample-B02'})))
                self.finished = True
                observation = {'accepted': True}
        elif role == 'refuse':
            self.contract_satisfied = self.decline
            self.finished = True
        return SimpleNamespace(data={'step': payload['step'], 'observation': observation, 'finished': self.finished}, error=None)

    def model(self, path, body):
        assert path == '/inference/chat/completions'
        self.model_calls.append(body)
        if body.get('tools'):
            if self.decline:
                role, args = 'refuse', {}
            elif self.changed and 'sample-A01' in self.cart:
                role, args = 'cart', {'id': 'sample-A01', 'quantity': 0}
            elif not self.cart:
                role, args = 'cart', {'id': 'sample-B02' if self.changed or self.preferred else 'sample-A01', 'quantity': 1}
            else:
                role, args = 'buy', {}
            message = {'tool_calls': [{'id': 'scripted-' + str(len(self.model_calls)), 'type': 'function',
                         'function': {'name': self.names[role], 'arguments': json.dumps(args)}}]}
        else:
            content = body['messages'][0]['content']
            prompt = content if isinstance(content, str) else content[0]['text']
            answer = {}
            if prompt.startswith(self.ns['BRIEF'][:60]):
                answer = {'items': [{'name': 'sample products', 'shelf': 'sample products', 'quantity': 1,
                                    'musts': ['surface: ' + ('absent' if self.decline else 'smooth')]}],
                          'roles': self.names, 'pick': 'any', 'messages': (1 if self.procedure else 0) + (3 if self.preferences else 0),
                          'steps': 35, 'buy_args': {}, 'refuse_args': {}, 'process':
                          [{'step': 'Explain the checks before purchasing', 'when': 'order', 'messages': 1, 'cart_dependent': False}] if self.procedure else []}
            elif prompt.startswith(self.ns['SHAPE'][:60]):
                answer = {'rows': 'products', 'count': 'count', 'price': 'price', 'stock': 'stock'}
            elif prompt.startswith(self.ns['PLAN'][:60]):
                answer = {'ask': [{'item': 1, 'label': key} for key in ('uniform', 'unvarying', 'marker')]}
            elif prompt.startswith(self.ns['HEAR'][:60]):
                answer = {'needs': [{'item': 1, 'need': 'marker: second', 'property_quote': 'marker'}] if self.last_answer == 'For marker, I require second.' else []}
            elif prompt.startswith(self.ns['CHECK'][:60]):
                answer = {'ok': not self.decline, 'problems': [], 'better': []}
            elif prompt.startswith('A shopping agent is about to'):
                if self.talk:
                    import re
                    lines = [int(x) for x in re.findall(r'(?m)^(\d+)\. step \d+: message to the shopper', prompt)]
                    answer = {'steps': [{'duty': 1, 'step': 'Explain the checks before purchasing', 'done': True, 'lines': lines}], 'call': None}
                else:
                    answer = {'steps': [{'duty': 1, 'done': False, 'lines': []}], 'call': {'tool': self.names['ask'], 'args': {'text': 'Check the supplied details before committing.'}}}
            elif prompt.startswith(self.ns['NOTE'][:60]):
                answer = {'kind': 'change', 'sold_out': self.changed, 'gone': [], 'summary': 'Replace the unavailable cart line.' if self.changed else ''}
            elif prompt.startswith(self.ns['SWAP'][:60]):
                answer = {'fits': [{'id': 'sample-B02', 'price': 2}]}
            elif prompt.startswith(self.ns['LINK_CHECK'][:60]):
                answer = {'match': True, 'missing': []}
            else:
                self.unknown_prompts.append(prompt[:90])
            message = {'content': json.dumps(answer)}
        return SimpleNamespace(data={'choices': [{'message': message}], 'usage': {'cost': 0.00001}}, error=None)

    def run(self):
        self.ns['_env'] = SimpleNamespace(post_verbose=self.env)
        self.ns['_llm_within'] = lambda _: SimpleNamespace(post_verbose=self.model)
        return self.ns['agent_main'](self.problem())


class IntegratedTests(unittest.TestCase):
    def test_interview_runs_through_entry_point_and_preserves_procedure_message(self):
        world = ScriptedWorld(load(), procedure=True, preferences=True)
        trace = world.run()
        self.assertTrue(world.finished, (trace[-8:], world.unknown_prompts))
        self.assertTrue(world.contract_satisfied)
        self.assertTrue(world.preferred and world.talk)
        self.assertEqual(world.cart, {'sample-B02': 1})
        self.assertTrue(any(r.get('kind') == 'question_focus' for r in trace))
        self.assertTrue(any(r.get('kind') == 'procedure_complete' for r in trace))
        self.assertLessEqual(sum(a['name'] == world.names['ask'] for a in world.actions), 4)

    def test_regular_environment_actions_are_available_to_procedure_review(self):
        ns = load()
        e = procedure_episode(ns)
        e.process = []
        e.roles['open'] = 'read_record'
        e.step = 0
        ns['_env'] = SimpleNamespace(post_verbose=lambda *args: SimpleNamespace(data={
            'step': 1, 'observation': {'id': 'record-001', 'title': 'observed', 'attributes': {'field': 'value', 'other': 'value'}}, 'finished': False}, error=None))
        e.act('read_record', {'id': 'record-001'})
        self.assertEqual(len(e.done_log), 1)
        self.assertEqual(e.done_steps, [1])

    def test_full_entry_point_with_ordinary_and_procedural_orders(self):
        for procedure in (False, True):
            with self.subTest(procedure=procedure):
                world = ScriptedWorld(load(), procedure=procedure)
                trace = world.run()
                self.assertTrue(world.finished, (trace[-8:], world.unknown_prompts))
                self.assertTrue(world.contract_satisfied)
                self.assertFalse([r for r in trace if r.get('kind') in ('crash', 'soft_crash')])
                self.assertEqual(world.cart, {'sample-A01': 1})
                if procedure:
                    self.assertTrue(world.talk)
                    self.assertTrue(any(r.get('kind') == 'procedure_complete' for r in trace))

    def test_full_entry_point_stockout_replacement(self):
        world = ScriptedWorld(load(), stockout=True)
        trace = world.run()
        self.assertTrue(world.finished, (trace[-8:], world.unknown_prompts))
        self.assertTrue(world.contract_satisfied)
        self.assertEqual(world.cart, {'sample-B02': 1})
        self.assertTrue(any(r.get('kind') == 'stockout_scope' for r in trace))

    def test_correct_decline_and_renamed_tools(self):
        world = ScriptedWorld(load(), decline=True, prefix='renamed_')
        trace = world.run()
        self.assertTrue(world.finished, (trace[-8:], world.unknown_prompts))
        self.assertTrue(world.contract_satisfied)
        self.assertEqual(world.actions[-1]['name'], world.names['refuse'])


class RealTransportTests(unittest.TestCase):
    def test_real_proxy_client_counts_one_http_402_and_keeps_binding_key(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from src.agent.proxy_client import ProxyClient
        ns = load()
        e = episode(ns)
        with TemporaryDirectory() as directory, patch.dict('os.environ', {}, clear=True):
            client = ProxyClient(proxy_url='http://offline.invalid', api_key='synthetic-test-token',
                                 timeout=1, max_retries=1,
                                 inference_stats_file=str(Path(directory)/'usage.jsonl'))
            ns['_llm_within'] = lambda _: client
            response = SimpleNamespace(status_code=402, text='synthetic exhausted credit',
                                       json=lambda: {'error': {'code': 402, 'message': 'synthetic exhausted credit'}})
            with patch('src.agent.proxy_client.requests.post', return_value=response) as post, \
                 patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')) as network:
                self.assertIsNone(e.think([], [ns['LIGHT'], ns['SPARE']], 1000))
                self.assertEqual(post.call_count, 1)
                self.assertEqual(e.calls, 1)
                self.assertTrue(e.no_credit)
                self.assertEqual(post.call_args.kwargs['headers']['Authorization'], 'Bearer synthetic-test-token')
                network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
