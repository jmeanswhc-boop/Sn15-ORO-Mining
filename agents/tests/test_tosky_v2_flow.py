"""Complete simulated flows through the real proxy client; no real HTTP is allowed."""
import json
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.proxy_client import ProxyClient
from test_tosky_v2 import agent, OfflineCase, cart_proof


class CompleteFlowTests(OfflineCase):
    def simulate(self, change=False, infeasible=False):
        tools = []
        for name, props in [('find', {'query': {'type': 'string'}}), ('read', {'id': {'type': 'string'}}),
                            ('basket', {'id': {'type': 'string', 'description': 'Observed product ID'}, 'quantity': {'type': 'integer'}}),
                            ('view', {}), ('buy', {}), ('decline', {})]:
            tools.append({'type': 'function', 'function': {'name': name, 'description': name,
                'parameters': {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}}})
        query = 'Find one equipment item with a satin surface. Spend at most 100 cents. No shopper messages. Finish within 20 steps.'
        data = {'environment': {'binding': {'session_id': 'offline-session', 'route': 'opaque-route', 'nonce': 'opaque-nonce'},
                               'policy_view': {'query': query, 'tools': tools, 'max_steps': 20, 'finished': False}}}
        records = {'one': {'id': 'one', 'category': 'equipment', 'name': 'Equipment item', 'price': 12,
                           'in_stock': True, 'attributes': {'surface': 'satin'}},
                   'other': {'id': 'other', 'category': 'equipment', 'name': 'Other equipment item', 'price': 15,
                             'in_stock': True, 'attributes': {'surface': 'gloss'}}}
        if infeasible:
            records['one']['attributes']['surface'] = 'gloss'
        state = {'searched': False, 'opened': False, 'opened_ids': [], 'cart': {}, 'buy_attempts': 0, 'finished': False}
        world_events, model_events, unknown = [], [], []
        brief = {'items': [{'name': 'equipment', 'shelf': 'equipment', 'quantity': 1, 'musts': ['surface: satin']}],
                 'budget': 100, 'budget_unit': 'cents', 'budget_scope': 'total', 'in_stock_only': True,
                 'pick': 'any', 'pick_measure': '', 'messages': 0, 'steps': 20, 'lookups': [], 'rules': [], 'wishes': [],
                 'ask_first': '', 'process': [], 'quiet_decline': True, 'fits': [],
                 'roles': {'search': 'find', 'open': 'read', 'offer': None, 'compare': None, 'ask': None,
                           'cart': 'basket', 'cart_view': 'view', 'buy': 'buy', 'refuse': 'decline'},
                 'search_words': {'tool': 'find', 'field': 'query', 'args': {}}, 'buy_args': {}, 'refuse_args': {}}
        audit = {'complete': True, 'procedures_complete': True, 'constraints': [
            {'item': 1, 'previous': 'surface: satin', 'need': 'surface: satin', 'relation': 'literal', 'source': 'with a satin surface'}],
                 'add': [], 'process': []}
        def response(content):
            return SimpleNamespace(status_code=200, text=json.dumps(content), json=lambda: content)
        def http(url, **kwargs):
            body = kwargs['json']
            if url.endswith('/environment/call'):
                self.assertNotIn('Authorization', kwargs.get('headers', {}))
                self.assertFalse(state['finished']);self.assertEqual(body['route'], 'opaque-route');self.assertEqual(body['nonce'], 'opaque-nonce')
                self.assertEqual(body['step'], len(world_events)+1)
                world_events.append(body)
                name, args = body['action']['name'], body['action']['args']
                if name == 'find':
                    state['searched'] = True
                    obs = {'results': [{k: v for k, v in r.items() if k != 'attributes'} for r in records.values()], 'total': 2, 'next': None}
                elif name == 'read':
                    state['opened'] = True;obs = records[args['id']]
                    if args['id'] not in state['opened_ids']:state['opened_ids'].append(args['id'])
                elif name == 'basket':
                    self.assertIn(args['id'], records);state['cart'][args['id']] = args['quantity'];obs = {'id': args['id'], 'quantity': args['quantity']}
                elif name == 'view':
                    obs = {'lines': [{'id': ident, 'quantity': qty, 'price': records[ident]['price']} for ident, qty in state['cart'].items()]}
                elif name == 'buy':
                    self.assertEqual(state['cart'], {'one': 1});state['buy_attempts'] += 1
                    if change and state['buy_attempts'] == 1:
                        obs = {'accepted': False, 'notice': ['The budget is now 120 cents.']}
                    else:state['finished'] = True;obs = {'accepted': True}
                elif name == 'decline' and infeasible:
                    self.assertEqual(state['cart'], {});state['finished'] = True;obs = {'accepted': True}
                else:
                    raise AssertionError('An unsupported final action was proposed: '+name)
                return response({'step': body['step'], 'finished': state['finished'], 'observation': obs})
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
                message = {'tool_calls': [{'id': 'synthetic-call', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}
            else:
                if content.startswith(agent.BRIEF_PROMPT):answer = brief
                elif content.startswith('Audit derived shopping constraints'):answer = audit
                elif content.startswith('Review a shopping decision'):answer = cart_proof()
                elif content.startswith('Audit a proposed no-purchase ending'):
                    answer = {'complete': True, 'source_complete': True, 'decline_supported': True,
                              'impossible_items': [1], 'fits': [], 'search_steps': [1], 'omitted_assessed': True,
                              'reason': 'The only opened candidate has gloss rather than the required satin surface; the other search result also has gloss.',
                              'checked': [{'id': ident, 'item': 1, 'need': 'surface: satin', 'status': 'mismatch',
                                           'evidence': [{'id': ident, 'scope': 'attribute', 'key': 'surface', 'value': 'gloss'}]}
                                          for ident in state['opened_ids']]}
                elif content.startswith(agent.AUDIT_PROMPT):answer = {'fits': {}, 'open': [], 'reason': 'The available equipment has gloss, not satin.'}
                elif content.startswith('Identify financial fields'):answer = {'price': 'price', 'stock': 'in_stock'}
                elif content.startswith(agent.SHAPE_PROMPT):answer = {'rows': 'results', 'count': 'total', 'next': 'next', 'next_arg': None, 'price': 'price', 'stock': 'in_stock'}
                elif content.startswith(agent.NOTE_PROMPT):answer = {'kind': 'change', 'budget': 120, 'add': [], 'drop': [], 'gone': [], 'sold_out': False, 'price_moved': False, 'summary': 'Budget is now 120 cents.'}
                else:
                    unknown.append(content[:250]);answer = {}
                message = {'content': json.dumps(answer)}
            return response({'usage': {'cost': .000001}, 'choices': [{'message': message, 'finish_reason': 'stop'}]})
        client = ProxyClient(api_key='offline-test-token', max_retries=1, timeout=1)
        made = []
        original_episode = agent.Episode
        def make_episode(payload):
            ep = original_episode(payload);made.append(ep);return ep
        with patch('src.agent.proxy_client.requests.post', side_effect=http), \
             patch('src.agent.proxy_client.requests.get', side_effect=AssertionError('Unexpected GET')), \
             patch.object(agent, 'model_client', return_value=client), patch.object(agent, 'ENV_CLIENT', client), patch.object(agent, 'Episode', side_effect=make_episode), patch.object(agent.time, 'sleep'):
            trace = agent.agent_main(data)
        diagnostic = {'cart': made[0].cart_lines, 'ids': made[0].cart_ids(), 'price_key': made[0].price_key,
                      'price': made[0].price('one'), 'page': made[0].page_read('one'), 'rows': made[0].rows.get('one'),
                      'needs': made[0].live(), 'proof': made[0].validate_cart_proof(cart_proof(), ['one'])} if made else {}
        self.assertTrue(state['finished'], {'trace': trace[-8:], 'unknown': unknown, 'diagnostic': diagnostic})
        self.assertEqual(unknown, [])
        self.assertFalse(any(t.get('kind') == 'crash' for t in trace), trace)
        self.assertTrue(any(t.get('kind') == ('decline_review' if infeasible else 'evidence_review') and
                            t.get('supported' if infeasible else 'positive') for t in trace))
        return state, world_events, model_events, trace

    def test_full_entrypoint_search_read_cart_review_and_order(self):
        state, world, model, trace = self.simulate()
        self.assertEqual(state['buy_attempts'], 1)
        self.assertLessEqual(len(model), agent.CALL_CAP)

    def test_rejected_checkout_applies_budget_notice_and_reviews_before_retry(self):
        state, world, model, trace = self.simulate(change=True)
        self.assertEqual(state['buy_attempts'], 2)
        self.assertGreaterEqual(sum(t.get('kind') == 'evidence_review' and t.get('positive') for t in trace), 2)

    def test_supported_decline_can_finish_without_forcing_a_purchase(self):
        state, world, model, trace = self.simulate(infeasible=True)
        self.assertEqual(state['buy_attempts'], 0)
        self.assertEqual(world[-1]['action']['name'], 'decline')
