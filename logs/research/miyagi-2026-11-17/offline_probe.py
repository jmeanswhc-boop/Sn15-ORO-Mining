"""Synthetic probes of the archived source with all live transport replaced."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from agents.tests.test_icdi_r1 import load

SOURCE = ROOT / 'champions/Miyagi_v2_06af7499-f202-7cf9-8000-3c856a5cf41f/agent.py'


def fresh():
    ns = load(SOURCE)
    ns['time'] = SimpleNamespace(monotonic=lambda: 1000.0, time=lambda: 1000.0, sleep=lambda _: None)
    e = ns['Cairn']({'environment': {'binding': {'session_id': 'synthetic'},
        'policy_view': {'query': 'Choose a matte object.', 'max_steps': 40, 'tools': []}}})
    e.items = [ns['Item']('object', 'Objects', 1, ['finish: matte'])]
    e.brief = {'pick': 'any', 'buy_args': {}, 'refuse_args': {}}
    e.budget = 100
    return ns, e


def main():
    results = {}
    ns, e = fresh()
    e.cart_lines = {'synthetic-listing': 1}
    e.ask_json = lambda *a: {'ok': False, 'problems': ['finish conflicts']}
    first, second = e.check_cart(), e.check_cart()
    assert first and second == ''
    results['negative_review_second_call'] = {'first_blocks': bool(first), 'second_blocks': bool(second),
                                              'stored_verdict': e.verdicts[e.check_key(e.cart_ids())]}

    ns, e = fresh()
    e.cart_lines = {'synthetic-listing': 1}
    e.ask_json = lambda *a: {'explanation': 'No decision was returned'}
    held = e.check_cart()
    results['review_without_ok_field'] = {'blocks': bool(held), 'stored_verdict': e.verdicts[e.check_key(e.cart_ids())]}
    assert not held and e.verdicts[e.check_key(e.cart_ids())]

    ns, e = fresh()
    e.cart_lines = {'synthetic-listing': 1}
    e.ask_json = lambda *a: None
    results['missing_review'] = {'blocks': bool(e.check_cart()), 'cached_as_checked': bool(e.checked)}

    ns, e = fresh()
    e.cart_lines = {'synthetic-listing': 1}
    before = e.check_key(e.cart_ids())
    e.cart_lines['synthetic-listing'] = 3
    results['cart_quantity_cache_key'] = {'same_after_quantity_change': before == e.check_key(e.cart_ids())}

    ns, e = fresh()
    original = {'finish.type': 'matte', 'finish_type': 'glossy', 'width': 'wide'}
    folded = ns['_fold_twins'](original)
    results['conflicting_spellings'] = {'input': original, 'folded': folded}
    assert 'matte, glossy' in folded.values()

    ns, e = fresh()
    results['short_change_notice'] = {'lines': e.loose_lines({'notice': 'Budget now 20.'})}
    line = 'Please change the total budget to twenty dollars immediately.'
    e.ask_json = lambda *a: None
    e.take_note([line])
    results['unread_note_retried'] = {'still_visible': bool(e.loose_lines({'notice': line})), 'budget': e.budget}
    assert line in e.heard and e.budget == 100

    ns, e = fresh()
    e.items.append(ns['Item']('second object', 'Other Objects', 1, []))
    e.take_heard({'needs': [{'item': 99, 'need': 'finish: glossy'}]}, [])
    results['invalid_item_scope'] = {'last_item_requirements': e.items[-1].told}
    assert e.items[-1].told == ['finish: glossy']

    ns, e = fresh()
    e.process = [{'step': 'Obtain approval', 'when': 'order', 'messages': 1}]
    e.ask_json = lambda *a: {'steps': [{'step': 'Obtain approval', 'done': False, 'lines': []}], 'call': None}
    go, _ = e.gate('order')
    results['gate_missing_proof'] = {'allows': go, 'cached_as_satisfied': not e.gate_due('order')}
    assert go and not e.gate_due('order')

    ns, e = fresh()
    e.cart_lines = {'synthetic-listing': 1}
    e.roles = {'buy': 'order'}
    e.schemas = {'order': {'type': 'object', 'properties': {}}}
    e.gate = lambda stage: (False, 'Approval withheld')
    calls = []
    def act(name, args):
        calls.append(name)
        e.finished = True
        return {'finished': True, 'observation': {}}
    e.act = act
    e.finish()
    results['finish_after_failed_gate'] = {'actions': calls}
    assert calls == ['order']

    ns, e = fresh()
    e.roles = {'open': 'inspect', 'cart': 'basket'}
    e.schemas = {n: {'type': 'object', 'properties': {}} for n in ('inspect', 'basket')}
    e.offered = lambda: [{'function': {'name': name}} for name in e.schemas]
    e.ask_json = lambda *a: {'kind': 'change', 'budget': 20, 'summary': 'Budget revised'}
    calls = []
    def execute(name, args):
        calls.append(name)
        e.step += 1
        return {'step': e.step, 'finished': False,
                'observation': {'notice': 'Please change the total budget to twenty dollars immediately.'} if name == 'inspect' else {}}
    e.act = execute
    e.run_calls([{'id': str(i), 'function': {'name': name, 'arguments': '{}'}} for i, name in enumerate(('inspect', 'basket'))])
    results['queued_action_after_change'] = {'actions': calls, 'version': e.version, 'budget': e.budget}
    assert calls == ['inspect', 'basket'] and e.budget == 20

    ns, e = fresh()
    physical = []
    def throttle(path, body):
        physical.append(body)
        return SimpleNamespace(data=None, error={'status': 429})
    ns['_llm'] = SimpleNamespace(post_verbose=throttle)
    e.think([{'role': 'user', 'content': 'A synthetic inference request'}], [ns['LIGHT']], 100)
    results['retry_counting'] = {'logical_counter': e.calls, 'physical_requests': len(physical)}
    assert len(physical) > e.calls

    ns, e = fresh()
    e.cost = 0.077
    ns['_llm'] = SimpleNamespace(post_verbose=lambda *a: SimpleNamespace(data={
        'choices': [{'message': {'content': 'synthetic'}}], 'usage': {'cost': 0.02}}, error=None))
    e.call_model(ns['LIGHT'], [{'role': 'user', 'content': 'synthetic'}], 100, None)
    results['cost_admission'] = {'before': 0.077, 'after': e.cost, 'named_hard_limit': ns['SPEND_HARD']}
    assert e.cost > ns['SPEND_HARD']

    ns, e = fresh()
    e.cost = 0.0
    engine = ns['BodyAsk'](e)
    engine.brief()
    item = engine.items[0]
    item.order = ['finish', 'surface texture']
    message = {'kind': 'list', 'parts': [(item, item.order)]}
    engine.think = lambda *a: {'kind': 'unread'}
    ns['SendoAsk'].hear(engine, message, 'Which property?', '')
    engine.think = lambda *a: None
    ns['SendoAsk'].hear(engine, message, 'Which property?', 'Please use a rough surface.')
    results['unread_answer_bridge'] = {'engine_record': engine.record, 'agent_requirements': e.needs_now()}
    assert engine.record and e.needs_now() == [(1, 'finish: matte')]

    path = Path(__file__).with_name('offline-probes.json')
    path.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
