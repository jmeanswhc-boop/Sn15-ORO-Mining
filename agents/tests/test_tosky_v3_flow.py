"""Controlled full entrypoint flows through ProxyClient, without live HTTP."""
import json
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.proxy_client import ProxyClient
from test_tosky_v3 import agent, baseline, OfflineCase


class EntrypointFlowTests(OfflineCase):
    def simulate(self, module=agent, *, disputed=False, malformed=False, change=False, infeasible=False):
        tools = []
        for name, props in [('find', {'query': {'type': 'string'}}), ('read', {'id': {'type': 'string'}}),
                            ('basket', {'id': {'type': 'string'}, 'quantity': {'type': 'integer'}}),
                            ('view', {}), ('buy', {}), ('decline', {})]:
            tools.append({'type': 'function', 'function': {'name': name, 'description': name,
                'parameters': {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}}})
        query = 'Find one equipment item with a satin surface. Spend at most 100 cents. No shopper messages. Finish within 20 steps.'
        if disputed:query = 'Choose the cheapest matching option. '+query
        data = {'environment': {'binding': {'session_id': 'offline-session', 'route': 'opaque-route', 'nonce': 'opaque-nonce'},
                               'policy_view': {'query': query, 'tools': tools, 'max_steps': 20, 'finished': False}}}
        records = {'one': {'id': 'one', 'category': 'equipment', 'name': 'Equipment item', 'price': 12,
                           'in_stock': True, 'attributes': {'surface': 'satin'}},
                   'other': {'id': 'other', 'category': 'equipment', 'name': 'Other equipment item', 'price': 15,
                             'in_stock': True, 'attributes': {'surface': 'gloss'}}}
        if infeasible:
            for record in records.values():record['price'] = 150
        state = {'searched': False, 'opened': False, 'cart': {}, 'buys': 0, 'finished': False,
                 'initial_rejection': False, 'corrective_calls': 0, 'review_response_sent': False}
        world_events, model_events, unknown, made = [], [], [], []
        brief = {'items': [{'name': 'equipment', 'shelf': 'equipment', 'quantity': 1, 'musts': ['surface: satin']}],
                 'budget': 100, 'budget_unit': 'cents', 'budget_scope': 'total', 'in_stock_only': True,
                 'pick': 'cheapest' if disputed else 'any', 'pick_measure': '', 'messages': 0, 'steps': 20, 'lookups': [], 'rules': [], 'wishes': [],
                 'ask_first': '', 'process': [], 'quiet_decline': True, 'fits': [],
                 'roles': {'search': 'find', 'open': 'read', 'offer': None, 'compare': None, 'ask': None,
                           'cart': 'basket', 'cart_view': 'view', 'buy': 'buy', 'refuse': 'decline'},
                 'search_words': {'tool': 'find', 'field': 'query', 'args': {}}, 'buy_args': {}, 'refuse_args': {}}
        def reply(content):
            return SimpleNamespace(status_code=200, text=json.dumps(content), json=lambda: content)
        def http(url, **kwargs):
            body = kwargs['json']
            if url.endswith('/environment/call'):
                self.assertNotIn('Authorization', kwargs.get('headers', {}));self.assertFalse(state['finished'])
                self.assertEqual(body['route'], 'opaque-route');self.assertEqual(body['nonce'], 'opaque-nonce')
                self.assertEqual(body['step'], len(world_events)+1);world_events.append(body)
                name, args = body['action']['name'], body['action']['args']
                if name == 'find':
                    state['searched'] = True
                    obs = {'results': [{k: v for k, v in r.items() if k != 'attributes'} for r in records.values()], 'total': 2, 'next': None}
                elif name == 'read':state['opened'] = True;obs = records[args['id']]
                elif name == 'basket':state['cart'][args['id']] = args['quantity'];obs = {'id': args['id'], 'quantity': args['quantity']}
                elif name == 'view':obs = {'lines': [{'id': ident, 'quantity': qty, 'price': records[ident]['price']} for ident, qty in state['cart'].items()]}
                elif name == 'buy':
                    self.assertFalse(infeasible);self.assertEqual(state['cart'], {'one': 1});state['buys'] += 1
                    if change and state['buys'] == 1:obs = {'accepted': False, 'notice': ['The budget is now 120 cents.']}
                    else:state['finished'] = True;obs = {'accepted': True}
                elif name == 'decline' and infeasible:state['finished'] = True;obs = {'accepted': True}
                else:raise AssertionError('Unsupported world action: '+name)
                return reply({'step': body['step'], 'finished': state['finished'], 'observation': obs})
            self.assertTrue(url.endswith('/inference/chat/completions'))
            self.assertEqual(kwargs.get('headers', {}).get('Authorization'), 'Bearer offline-test-token')
            model_events.append(body)
            content = '\n'.join(m.get('content', '') if isinstance(m.get('content'), str) else
                                '\n'.join(p.get('text', '') for p in (m.get('content') or []) if isinstance(p, dict)) for m in body['messages'])
            if body.get('tools'):
                if not state['searched']:name, args = 'find', {'query': 'equipment'}
                elif not state['opened']:name, args = 'read', {'id': 'one'}
                elif infeasible:name, args = 'decline', {}
                elif not state['cart']:name, args = 'basket', {'id': 'one', 'quantity': 1}
                else:name, args = 'buy', {}
                message = {'tool_calls': [{'id': 'controlled-call', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}
            else:
                if content.startswith(module.BRIEF_PROMPT):answer = brief
                elif content.startswith('Audit the request'):answer = {'complete': True, 'missing': []}
                elif content.startswith(module.CHECK_PROMPT):
                    negative = disputed and state['cart'] and not state['review_response_sent']
                    if negative:state['initial_rejection'] = True
                    answer = {'ok': not negative, 'problems': ['The observed surface is disputed.'] if negative else [], 'better': []}
                elif content.startswith('Resolve a disputed shopping review'):
                    state['corrective_calls'] += 1;packet = json.loads(content.split('SNAPSHOT:\n', 1)[1])
                    candidates = []
                    for candidate in packet['candidates']:
                        fields = [(eid, f) for eid, f in packet['facts'].items() if f[0] == candidate['cid']]
                        kind = next(eid for eid, f in fields if f[2] == 'category')
                        surface = next(eid for eid, f in fields if f[1:3] == ['attribute', 'surface'])
                        candidates.append({'cid': candidate['cid'], 'kind': 'match', 'kind_evidence': [kind],
                            'needs': [{'rid': r['rid'], 'state': 'met', 'same_scope': True, 'evidence': [surface]} for r in packet['requirements']]})
                    answer = {'revision': packet['revision'], 'candidates': candidates}
                    if malformed and state['corrective_calls'] == 1:answer['candidates'][0]['needs'][0]['evidence'] = 'The surface matches.'
                    else:state['review_response_sent'] = True
                elif content.startswith(module.AUDIT_PROMPT):answer = {'fits': {}, 'open': [], 'reason': 'All observed options exceed the budget.'}
                elif content.startswith('Identify financial fields'):answer = {'price': 'price', 'stock': 'in_stock'}
                elif content.startswith(module.SHAPE_PROMPT):answer = {'rows': 'results', 'count': 'total', 'next': 'next', 'next_arg': None, 'price': 'price', 'stock': 'in_stock'}
                elif content.startswith(module.NOTE_PROMPT):answer = {'kind': 'change', 'budget': 120, 'add': [], 'drop': [], 'gone': [], 'sold_out': False, 'price_moved': False, 'summary': 'Budget is now 120 cents.'}
                else:unknown.append(content[:250]);answer = {}
                message = {'content': json.dumps(answer)}
            return reply({'usage': {'cost': .000001}, 'choices': [{'message': message, 'finish_reason': 'stop'}]})
        client = ProxyClient(api_key='offline-test-token', max_retries=1, timeout=1)
        original = module.Episode
        def make(payload):
            ep = original(payload);made.append(ep);return ep
        with patch('src.agent.proxy_client.requests.post', side_effect=http), \
             patch('src.agent.proxy_client.requests.get', side_effect=AssertionError('Unexpected GET')), \
             patch.object(module, 'model_client', return_value=client), patch.object(module, 'ENV_CLIENT', client), \
             patch.object(module, 'Episode', side_effect=make), patch.object(module.time, 'sleep'):
            trace = module.agent_main(data)
        self.assertEqual(unknown, [])
        self.assertTrue(state['finished'], {'state': state, 'blocks': made[0].review_block_counts,
            'disputes': made[0].dispute_requests, 'verdicts': made[0].verdicts,
            'trace': [t for t in trace if t.get('kind') not in ('turn', 'inference_budget')][-22:]})
        self.assertFalse(any(e.get('kind') == 'crash' for e in trace))
        self.assertEqual(made[0].calls, len(model_events));self.assertEqual(made[0].inference_budget.pending, 0)
        self.assertLessEqual(made[0].inference_budget.actual+made[0].inference_budget.uncertain, made[0].inference_budget.limit)
        return state, world_events, model_events, trace

    def test_healthy_full_flow_matches_v4_requests_and_world_actions(self):
        runs = [self.simulate(module) for module in (baseline, agent)]
        self.assertEqual(runs[0][1], runs[1][1]);self.assertEqual(runs[0][2], runs[1][2])
        self.assertEqual(runs[1][0]['corrective_calls'], 0)

    def test_negative_review_is_corrected_before_successful_checkout(self):
        state, world, _, trace = self.simulate(disputed=True)
        self.assertTrue(state['initial_rejection']);self.assertEqual(state['corrective_calls'], 1)
        self.assertEqual(world[-1]['action']['name'], 'buy')
        self.assertTrue(any(t.get('kind') == 'review_boundary' and t['status'] == 'APPROVED' for t in trace))

    def test_malformed_review_gets_format_only_retry_and_finishes(self):
        state, world, _, trace = self.simulate(disputed=True, malformed=True)
        self.assertEqual(state['corrective_calls'], 2);self.assertEqual(world[-1]['action']['name'], 'buy')
        self.assertTrue(any(t.get('kind') == 'review_boundary_response' and t['status'] == 'FORMAT_ERROR' for t in trace))

    def test_rejected_checkout_notice_keeps_existing_revalidation(self):
        state, _, _, trace = self.simulate(change=True)
        self.assertEqual(state['buys'], 2)
        self.assertGreaterEqual(sum(t.get('kind') == 'cart_check' and t.get('ok') is True for t in trace), 2)

    def test_no_affordable_candidate_keeps_the_original_decline_route(self):
        state, world, _, _ = self.simulate(infeasible=True)
        self.assertEqual(state['buys'], 0);self.assertEqual(world[-1]['action']['name'], 'decline')
