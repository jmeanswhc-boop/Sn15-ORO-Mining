"""Counterexamples from public observations; no world or inference calls."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from agents.tests.test_tosky_v4 import load

HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'logs/research/tosky-v5-local-review-2026-11-15'
RUN = 'local-6dfac68554764b9e8f2f025d9f0a1777'


def snapshot(task, stop=None, source='agents/Tosky_v6.py'):
    d = json.loads((EVIDENCE / 'tasks' / RUN / (task + '.json')).read_text())
    problems = [json.loads(line) for line in (ROOT / 'LV_logs' / RUN / 'problems.jsonl').read_text().splitlines() if line.strip()]
    p = next(p for p in problems if p['query'] == d['request'])
    ns = load(ROOT / source)
    e = ns['Cairn']({'environment': {'binding': {'offline': True}, 'policy_view': copy.deepcopy(p['environment']['policy_view'])}})
    e.brief = copy.deepcopy(next(x['brief'] for x in d['diagnostics'] if x.get('kind') == 'brief'))
    e.roles = e.brief['roles']
    e.items = [ns['Item'](x['name'], x.get('shelf', ''), x['quantity'], x['musts']) for x in e.brief['items']]
    e.budget, e.msg_limit = e.brief['budget'], e.brief['messages']
    e.id_key, e.price_key, e.stock_key = 'item_pid', 'price', 'in_stock'
    tool = None
    for event in d['world']:
        if stop is not None and event.get('step', 0) > stop:
            break
        if event['kind'] == 'action':
            tool, e.step = event['tool'], event['step']
        elif event['kind'] == 'observation':
            e.absorb(e.unspelled(tool, event['observation']))
    return ns, e, d, p


def stage_one():
    out = {}
    for source in ('agents/Tosky_v5.py', 'agents/Tosky_v6.py'):
        _, e, _, _ = snapshot('t-d4e2bc9e66', 13, source)
        out[source] = {'skirt': {i: {'verdict': e.key_verdict(i, 'style: sweet'), 'unfit': e.unfit(i)}
                                for i in ('p-1f4aa8507466', 'p-338c87630a54')}}
        _, e, d, _ = snapshot('t-9ab5c575f5', 15, source)
        e.cart_lines, e.cart_touched = {'p-028af8354258': 1}, True
        e.cart_at, e.cart_seen_at = 13, 15
        e.cart_view = next(x['observation'] for x in d['world'] if x['kind'] == 'observation' and x['step'] == 15)
        out[source]['alternative_cart_faults'] = e.line_faults()
        _, e, _, _ = snapshot('t-abf1c45838', source=source)
        lost = 'p-73f0e390ac95'
        out[source]['clock_recovery_pool'] = e.recovery_pool([lost])
        out[source]['clock_predicate'] = e.key_verdict('p-180b10b52cb0', 'shape: square')
    after = out['agents/Tosky_v6.py']
    assert all(v['verdict'][0] is None and not v['unfit'] for v in after['skirt'].values())
    assert not after['alternative_cart_faults']
    assert 'p-180b10b52cb0' in after['clock_recovery_pool']
    out['limits'] = 'Fixed observations only. Ambiguous evidence reaches semantic review; no new task score or inference outcome is asserted.'
    return out


def stage_two():
    out = {}
    for task in ('t-069a67a0c4', 't-9ab5c575f5'):
        _, e, d, _ = snapshot(task, 34 if task == 't-069a67a0c4' else 35)
        e.max_steps = 38
        e.cart_touched = True
        e.cap_seen = len(e.full)
        e.cart_at, e.cart_seen_at = e.step, 23
        e.cart_view = {'lines_items': [{'item_pid': 'p-0ec67862ffe2', 'qty': 1}]}
        e.cart_lines = {} if task == 't-069a67a0c4' else {'p-1c437362968f': 1}
        if e.cart_lines:
            e.unavailable.update(e.cart_lines)
        calls = []
        def act(name, args):
            calls.append(name)
            e.step += 1
            e.finished = name == e.roles['refuse']
            return {'observation': {}, 'step': e.step, 'finished': e.finished}
        e.act = act
        repair = e.finish(again=True)
        if not e.finished:
            e.finish()
        assert calls == [e.roles['refuse']] and e.finished
        out[task] = {'repair_message': repair, 'actions': calls, 'finished': e.finished,
                     'limit': 'Scripted terminal acknowledgement. This proves control-flow completion, not a correct decline score.'}
    return out


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else '01'
    result = stage_two() if stage == '02' else stage_one()
    (HERE / (stage + '-public-replay.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
