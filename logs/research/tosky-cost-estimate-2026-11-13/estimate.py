"""Offline cost sensitivity for unchanged TOSKY; never sends paid requests."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
AUDIT = ROOT.parent / 'tosky-framework-audit-2026-11-13'
sys.path[:0] = [str(PROJECT), str(AUDIT)]
import probe_flows

SOURCE = PROJECT / 'agents/TOSKY.py'
EXPECTED = '31b083e8ac95bc6863a4a01f9768a61879914fed4eb9116ea0f27b394d825a18'


def question_flow(preferences):
    """Faithful scripted interpretations expose actual controller call counts."""
    module = probe_flows.load()
    payload = probe_flows.problem()
    policy = payload['environment']['policy_view']
    policy.update(max_steps=38, query='Buy one travel bag with a matte surface for no more than 50 credits. You may send at most 3 messages.')
    roles = {'search': 'find_catalog', 'open': 'read_item', 'offer': 'read_offer', 'compare': 'compare_items',
             'message': 'ask_user', 'cart': 'change_cart', 'cart_view': 'inspect_cart', 'checkout': 'order_now', 'decline': 'stop_shopping'}
    item = {'id': 'A', 'kind': 'travel bag', 'price': 10, 'stock': True,
            'features': {'surface': 'matte', 'color': 'blue', 'size': 'large'}}
    state = {'cart': {}, 'finished': False, 'evaluations': 0, 'messages': 0, 'reply': '', 'property': ''}
    made, worlds, calls = [], [], []

    def environment(path, body):
        assert path == '/environment/call' and body['step'] == len(worlds) + 1 and not state['finished']
        worlds.append(deepcopy(body))
        name, args = body['action']['name'], body['action']['args']
        if name == 'find_catalog':
            observation = {'results': [{k: v for k, v in item.items() if k != 'features'}], 'total': 1, 'next': None}
        elif name == 'read_item':
            assert args['id'] == 'A'
            observation = item
        elif name == 'read_offer':
            observation = {k: v for k, v in item.items() if k != 'features'}
        elif name == 'ask_user':
            prop = made[0].pending_question['property']
            state['messages'] += 1
            state['property'] = prop
            state['reply'] = ('I require ' + prop + ': ' + item['features'][prop]) if preferences else 'I have no preference about ' + prop
            observation = {'reply': state['reply']}
        elif name == 'change_cart':
            state['cart'][args['id']] = args['quantity']
            observation = {'changed': True}
        elif name == 'inspect_cart':
            observation = {'lines': [{'id': k, 'quantity': q, 'price': 10} for k, q in state['cart'].items()]}
        elif name in ('order_now', 'stop_shopping'):
            state['finished'] = True
            observation = {'accepted': True}
        else:
            raise AssertionError(name)
        return SimpleNamespace(data={'step': body['step'], 'finished': state['finished'], 'observation': deepcopy(observation)}, error=None)

    def inference(path, body):
        assert path == '/inference/chat/completions'
        prompt = body['messages'][1]['content']
        ep = made[0]
        purpose = ''
        if prompt.startswith('Interpret this shopper request'):
            purpose = 'contract'
            answer = {'items': [{'role': 'travel bag', 'quantity': 1, 'hard': ['surface: matte'], 'soft': []}],
                      'budget': {'amount': 50, 'unit': 'credits', 'scope': 'total'}, 'objective': {'kind': 'any'},
                      'message_limit': 3, 'questions': [], 'obligations': [], 'relations': [], 'roles': roles}
        elif prompt.startswith('Choose exactly ONE world action'):
            intent = prompt.split('INTENT: ', 1)[1].split('\n', 1)[0]
            purpose = 'action ' + intent
            if intent == 'search':
                name, args = 'find_catalog', {'query': 'travel bag'}
            elif intent == 'open':
                name, args = ('read_offer' if 'current offer and availability' in prompt else 'read_item'), {'id': 'A'}
            elif intent == 'question':
                prop = ep.question_to_ask()['property']
                name, args = 'ask_user', {'text': 'What is your preference for ' + prop + '?'}
            elif intent == 'cart view':
                name, args = 'inspect_cart', {}
            elif intent == 'cart':
                name, args = 'change_cart', {'id': 'A', 'quantity': 1}
            elif intent == 'checkout':
                name, args = 'order_now', {}
            elif intent == 'decline':
                name, args = 'stop_shopping', {}
            else:
                raise AssertionError(intent)
            answer = {'tool': name, 'args': args}
        elif prompt.startswith('Read one world observation'):
            purpose = 'observation'
            name = ep.last_action['name']
            answer = {'cart_lines': None, 'changes': [], 'reply': None, 'approval': None, 'error': None}
            if name == 'find_catalog':
                answer['shape'] = {'rows_path': 'results', 'id_key': 'id', 'price_key': 'price', 'stock_key': 'stock',
                                   'count_path': 'total', 'next_path': 'next', 'price_unit': 'credits'}
            elif name == 'inspect_cart':
                answer['cart_lines'] = [{'id': k, 'quantity': q} for k, q in state['cart'].items()]
            elif name == 'ask_user':
                answer['reply'] = state['reply']
        elif prompt.startswith('Judge ONLY this listing'):
            purpose = 'candidate evidence'
            state['evaluations'] += 1
            c = ep.candidates[(1, 'A')]
            prefix = 'views.0.features.' if isinstance(c.detail, dict) and 'views' in c.detail else 'features.'
            matches = []
            for requirement in ep.slots[0].hard:
                prop, value = (s.strip() for s in requirement.split(':', 1))
                assert item['features'][prop] == value
                matches.append({'requirement': requirement, 'status': 'true', 'source': 'detail', 'path': prefix + prop, 'quote': value})
            answer = {'matches': matches, 'stock': {'status': 'true', 'source': 'row', 'path': 'stock', 'quote': 'true'},
                      'price': {'amount': 10, 'unit': 'credits', 'source': 'row', 'path': 'price', 'quote': '10'}}
        elif prompt.startswith('Suggest at most TWO focused shopper questions'):
            purpose = 'question discovery'
            answer = {'questions': [{'item': 1, 'property': p, 'why': 'An unsettled property of the requested item'} for p in ('color', 'size')]}
        elif prompt.startswith('Extract only changes explicitly stated'):
            purpose = 'shopper revision'
            prop = state['property']
            answer = {'add': [{'item': 1, 'requirement': prop + ': ' + item['features'][prop], 'quote': state['reply']}] if preferences else [],
                      'drop': [], 'budget': None, 'budget_unit': '', 'budget_quote': '',
                      'free': [] if preferences else [{'item': 1, 'property': prop}], 'refusal': False}
        else:
            raise AssertionError(prompt[:150])
        text = json.dumps(answer)
        calls.append({'model': body['model'], 'purpose': purpose, 'max_output_tokens': body['max_tokens'],
                      'prompt_characters_including_system': sum(len(m['content']) for m in body['messages']),
                      'scripted_reply_characters': len(text)})
        return SimpleNamespace(data={'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}, error=None)

    original = module.Tosky
    def create(data):
        ep = original(data)
        made.append(ep)
        return ep

    with patch.object(module, 'Tosky', side_effect=create), \
         patch.object(module, '_environment', SimpleNamespace(post_verbose=environment)), \
         patch.object(module, '_inference', SimpleNamespace(post_verbose=inference)):
        trace = module.agent_main(payload)
    assert not [row for row in trace if row.get('kind') == 'tosky_error'], trace
    assert state['messages'] == 2 and state['cart'] == {'A': 1}
    assert worlds[-1]['action']['name'] == 'order_now'
    assert made[0].revision == (2 if preferences else 0)
    return {'scenario': 'two_preference_replies' if preferences else 'two_indifferent_replies',
            'world_steps': len(worlds), 'messages': state['messages'], 'candidate_evaluations': state['evaluations'],
            'model_counts': dict(Counter(c['model'] for c in calls)),
            'purpose_counts': dict(Counter(c['purpose'] for c in calls)), 'calls': calls,
            'token_counts_measured': False, 'actual_billed_cost': None,
            'note': 'Scripted successful responses prove call counts for this flow only, not model accuracy or task-mix frequency.'}


def cost(sonnet, haiku, input_tokens=2000, sonnet_output=400, haiku_output=150):
    low = (Decimal(sonnet) * (Decimal(input_tokens) * 2 + Decimal(sonnet_output) * 10)
           + Decimal(haiku) * (Decimal(input_tokens) / 10 + Decimal(haiku_output) / 2)) / 1000000
    high = low * Decimal('1.1')
    return {'per_task_usd': [str(low), str(high)], 'thirty_tasks_usd': [str(low * 30), str(high * 30)],
            'ninety_tasks_usd': [str(low * 90), str(high * 90)]}


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    with patch('requests.sessions.Session.request', side_effect=AssertionError('Live HTTP prohibited')) as network, \
         patch.dict('os.environ', {'SANDBOX_MODEL': ''}):
        flows = [question_flow(False), question_flow(True)]
        network.assert_not_called()
    previous = json.loads((ROOT.parent / 'tosky-competitive-review-2026-11-13/review-evidence.json').read_text())
    old = previous['public_limit_flows']['flows']
    scenarios = []
    for row in old:
        if row['scenario'] not in ('healthy', 'read_procedure'):
            continue
        counts = row['summary']['model_distribution']
        scenarios.append({'name': row['scenario'], 'source': 'prior controlled full flow', 'counts': counts})
    for row in flows:
        scenarios.append({'name': row['scenario'], 'source': 'new controlled full flow', 'counts': row['model_counts']})
    for row in scenarios:
        row['cost_sensitivity'] = {}
        for label, tokens in [('compact', (1000, 250, 100)), ('reference_assumptions', (2000, 400, 150)), ('larger', (4000, 800, 300))]:
            row['cost_sensitivity'][label] = cost(row['counts'].get('anthropic/claude-sonnet-5.5', 0),
                row['counts'].get('anthropic/claude-haiku-5.5', 0), *tokens)

    output = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': EXPECTED,
              'rates_usd_per_million': {'haiku': {'input': ['0.10', '0.11'], 'output': ['0.50', '0.55']},
                'sonnet': {'input': ['2.00', '2.20'], 'output': ['10.00', '11.00']},
                'deepseek': {'input': ['0.2088', '0.30'], 'output': ['0.3096', '0.96']}},
              'pricing_checked': 'Public OpenRouter endpoint catalogs read through web this turn; matching November 13 archived rates.',
              'pricing_sources': ['https://openrouter.ai/api/v1/models/' + model + '/endpoints' for model in
                                  ('anthropic/claude-haiku-5.5', 'anthropic/claude-sonnet-5.5', 'deepseek/deepseek-v3.2')],
              'assumptions': {'compact': {'input_per_call': 1000, 'sonnet_output_per_call': 250, 'haiku_output_per_call': 100},
                              'reference_assumptions': {'input_per_call': 2000, 'sonnet_output_per_call': 400, 'haiku_output_per_call': 150},
                              'larger': {'input_per_call': 4000, 'sonnet_output_per_call': 800, 'haiku_output_per_call': 300}},
              'limitations': ['Scenario arithmetic, not measured provider tokens, a statistical forecast or confidence interval.',
                              'Thirty/ninety repetitions assume identical flow and token averages; actual task mixture is unknown.',
                              'No cache discount, backup, uncertain-charge, funding fee or infrastructure cost included.',
                              'Rates assume base context tiers and no SANDBOX_MODEL override.',
                              'Demand over the ORO cap describes attempted workload completion; the platform will block new calls before full completion.'],
              'scenarios': scenarios, 'new_offline_profiles': flows, 'paid_calls': 0, 'agent_modified': False}
    (ROOT / 'cost-estimate.json').write_text(json.dumps(output, indent=2) + '\n')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    print(json.dumps({'source_unchanged': True, 'paid_calls': 0, 'new_profiles': [{k: f[k] for k in
        ('scenario', 'world_steps', 'messages', 'candidate_evaluations', 'model_counts', 'purpose_counts')} for f in flows],
        'scenarios': scenarios}, indent=2))


if __name__ == '__main__':
    main()
