"""Controlled end-to-end framework flows; no live model, world or grading calls."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from probe_framework import ROOT, PROJECT, SOURCE, DIGEST, problem


def load():
    spec = importlib.util.spec_from_file_location('tosky_framework_flow_audit', SOURCE)
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module;spec.loader.exec_module(module)
    return module


def run(module, scenario):
    payload = problem();policy = payload['environment']['policy_view']
    policy['query'] = 'Buy one travel bag with a matte surface for no more than 50 credits. No shopper messages.'
    requirement = 'surface: matte'
    obligations = []
    if scenario == 'reference_dependency':
        policy['query'] = 'Buy one travel bag whose surface matches reference R, at most 50 credits. Read reference R before ordering. No shopper messages.'
        requirement = 'surface: same as reference R'
        obligations = [{'description': 'Read reference R before ordering', 'stage': 'before_order', 'kind': 'read'}]
    elif scenario == 'read_procedure':
        policy['query'] += ' After preparing the basket, read its selected item page before ordering.'
        obligations = [{'description': 'Read selected item A after preparing the basket', 'stage': 'before_order', 'kind': 'read'}]
    roles = {'search': 'find_catalog', 'open': 'read_item', 'offer': 'read_offer', 'compare': 'compare_items',
             'message': 'ask_user', 'cart': 'change_cart', 'cart_view': 'inspect_cart', 'checkout': 'order_now', 'decline': 'stop_shopping'}
    item = {'id': 'A', 'kind': 'travel bag', 'price': 10, 'stock': True, 'features': {'surface': 'matte'}}
    reference = {'id': 'R', 'kind': 'reference item', 'features': {'surface': 'matte'}}
    state = {'cart': {}, 'finished': False, 'searches': 0, 'evaluations': 0, 'reference_reads': 0}
    worlds, models, made = [], [], []
    def environment(path, body):
        assert path == '/environment/call' and body['route'] == 'opaque-route'
        assert body['step'] == len(worlds)+1 and not state['finished']
        worlds.append(deepcopy(body));name, args = body['action']['name'], body['action']['args']
        if name == 'find_catalog':
            state['searches'] += 1;observation = {'results': [{k: v for k, v in item.items() if k != 'features'}], 'total': 1, 'next': None}
        elif name == 'read_item':
            if args['id'] == 'R':state['reference_reads'] += 1;observation = reference
            else:assert args['id'] == 'A';observation = item
        elif name == 'read_offer':observation = {k: v for k, v in item.items() if k != 'features'}
        elif name == 'change_cart':state['cart'][args['id']] = args['quantity'];observation = {'changed': True}
        elif name == 'inspect_cart':observation = {'lines': [{'id': k, 'quantity': q, 'price': 10} for k, q in state['cart'].items()]}
        elif name in ('order_now', 'stop_shopping'):state['finished'] = True;observation = {'accepted': True}
        else:raise AssertionError(name)
        return SimpleNamespace(data={'step': body['step'], 'finished': state['finished'], 'observation': deepcopy(observation)}, error=None)
    def inference(path, body):
        assert path == '/inference/chat/completions'
        models.append(deepcopy(body));prompt = body['messages'][1]['content'];ep = made[0]
        if prompt.startswith('Interpret this shopper request'):
            answer = {'items': [{'role': 'travel bag', 'quantity': 1, 'hard': [requirement], 'soft': []}],
                      'budget': {'amount': 50, 'unit': 'credits', 'scope': 'total'}, 'objective': {'kind': 'any'},
                      'message_limit': 0, 'questions': [], 'obligations': obligations, 'relations': [], 'roles': roles}
        elif prompt.startswith('Choose exactly ONE world action'):
            intent = prompt.split('INTENT: ', 1)[1].split('\n', 1)[0]
            if intent == 'search':name, args = 'find_catalog', {'query': ['travel bag', 'bag travel', 'bag', 'travel'][min(state['searches'], 3)]}
            elif intent == 'open':name, args = ('read_offer' if 'current offer and availability' in prompt else 'read_item'), {'id': 'A'}
            elif intent == 'cart view':name, args = 'inspect_cart', {}
            elif intent == 'cart':name, args = 'change_cart', {'id': 'A', 'quantity': 1}
            elif intent == 'checkout':name, args = 'order_now', {}
            elif intent == 'decline':name, args = 'stop_shopping', {}
            elif intent == 'procedure':name, args = 'read_item', {'id': 'R' if scenario == 'reference_dependency' else 'A'}
            else:raise AssertionError(intent)
            answer = {'tool': name, 'args': args}
        elif prompt.startswith('Read one world observation'):
            name = ep.last_action['name'];answer = {'cart_lines': None, 'changes': [], 'reply': None, 'approval': None, 'error': None}
            if name == 'find_catalog':answer['shape'] = {'rows_path': 'results', 'id_key': 'id', 'price_key': 'price',
                'stock_key': 'stock', 'count_path': 'total', 'next_path': 'next', 'price_unit': 'credits'}
            if name == 'inspect_cart':answer['cart_lines'] = [{'id': k, 'quantity': q} for k, q in state['cart'].items()]
        elif prompt.startswith('Judge ONLY this listing'):
            state['evaluations'] += 1;c = ep.candidates[(1, 'A')]
            field_path = 'views.0.features.surface' if isinstance(c.detail, dict) and 'views' in c.detail else 'features.surface'
            match = {'requirement': requirement, 'status': 'true', 'source': 'detail', 'path': field_path, 'quote': 'matte'}
            if scenario == 'format_failure' and state['evaluations'] == 1:match['quote'] = 'The product seems suitable.'
            if scenario == 'reference_dependency':
                if not ep.references:match['status'] = 'unknown'
                else:match.update(reference_path='0.features.surface', reference_quote='matte')
            answer = {'matches': [match], 'stock': {'status': 'true', 'source': 'row', 'path': 'stock', 'quote': 'true'},
                      'price': {'amount': 10, 'unit': 'credits', 'source': 'row', 'path': 'price', 'quote': '10'}}
        elif prompt.startswith('Did this observed action actually fulfill'):
            answer = {'satisfied': True, 'quote': 'A'}
        else:raise AssertionError(prompt[:140])
        return SimpleNamespace(data={'usage': {'cost': .000001}, 'choices': [
            {'message': {'content': json.dumps(answer)}, 'finish_reason': 'stop'}]}, error=None)
    original = module.Tosky
    def create(data):
        ep = original(data);made.append(ep);return ep
    with patch.object(module, 'Tosky', side_effect=create), \
         patch.object(module, '_environment', SimpleNamespace(post_verbose=environment)), \
         patch.object(module, '_inference', SimpleNamespace(post_verbose=inference)):
        trace = module.agent_main(payload)
    terminal = next((w['action']['name'] for w in reversed(worlds) if w['action']['name'] in ('order_now', 'stop_shopping')), None)
    errors = [t for t in trace if t.get('kind') == 'tosky_error']
    assert not errors, errors
    if scenario == 'healthy':assert terminal == 'order_now' and state['cart'] == {'A': 1}
    if scenario == 'format_failure':assert terminal == 'stop_shopping' and state['evaluations'] == 1
    if scenario == 'reference_dependency':assert terminal == 'stop_shopping' and state['reference_reads'] == 0
    if scenario == 'read_procedure':assert sum(w['action']['name'] == 'read_item' for w in worlds) > 3 and terminal != 'order_now'
    return {'scenario': scenario, 'finished': state['finished'], 'terminal': terminal, 'steps': len(worlds),
            'model_requests': len(models), 'evaluations': state['evaluations'], 'reference_reads': state['reference_reads'],
            'requested_output_tokens_total': sum(b['max_tokens'] for b in models),
            'model_prompt_utf8_bytes_total': sum(len(b['messages'][1]['content'].encode()) for b in models),
            'world_actions': [w['action']['name'] for w in worlds],
            'model_distribution': {m: sum(b['model'] == m for b in models) for m in sorted({b['model'] for b in models})},
            'limitation': 'Controlled interpreter and world replies. No live task score or measured inference cost/latency.'}


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == DIGEST
    with patch('requests.sessions.Session.request', side_effect=AssertionError('Live network forbidden')) as network, \
         patch.dict('os.environ', {'SANDBOX_MODEL': ''}):
        module = load();results = [run(module, scenario) for scenario in ('healthy', 'format_failure', 'reference_dependency', 'read_procedure')]
        network.assert_not_called()
    (ROOT/'controlled-flows.json').write_text(json.dumps({'source_sha256': DIGEST, 'paid_calls': 0, 'results': results}, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':main()
