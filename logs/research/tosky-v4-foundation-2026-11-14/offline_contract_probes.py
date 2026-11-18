"""Exercise isolated public PIRATE methods with synthetic, offline dependencies.

These probes establish control-flow behavior, not benchmark scores. No archived
module is imported, no top-level agent code runs, and no network client is used.
"""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SOURCE = PROJECT/'champions/PIRATE_v1_06af6030-67a0-754e-8000-beacfd741adb/agent.py'


def extract():
    parsed = ast.parse(SOURCE.read_text())
    cairn = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Cairn')
    names = {'direct_order', 'finish', 'gate', 'think', 'call_model', 'scaled_same'}
    namespace = {'Any': object, 're': re, '_shape': lambda args, *_: args, '_clip': lambda value, *_: value,
                 '_dump': json.dumps, 'GATE_ROUNDS': 3, 'GATE': 'STAGE', 'LIGHT': 'light', 'HEAVY': 'heavy',
                 'ALL_CALLS': 60, 'CALL_FLOOR': 6.0, 'WAITS': (4, 10, 20, 35), 'WAIT_TOTAL': 45,
                 'time': SimpleNamespace(sleep=lambda _: None), 'random': SimpleNamespace(random=lambda: .5),
                 '_items': lambda value: value if isinstance(value, list) else [],
                 '_count': lambda value: int(value) if isinstance(value, (int, str)) else None}
    for node in cairn.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            isolated = copy.deepcopy(node)
            for arg in isolated.args.args:
                arg.annotation = None
            isolated.returns = None
            exec(compile(ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[])), str(SOURCE), 'exec'), namespace)
    return namespace


def cart_fixture():
    state = SimpleNamespace(roles={'buy':'buy'}, direct_tried=False, finished=False, broke=False,
                            brief={}, schemas={'buy':{}}, trace=[], buys=0, step=3,
                            cart_view={}, cart_seen_at=3, cart_at=2, cart_lines={'item':1},
                            items=[object()], full={}, cap_seen=0, cart_touched=True)
    state.left = lambda: 5
    state.category_key = lambda: 'category'
    state.cart_ids = lambda: ['item']
    state.over_lines = lambda: []
    state.short_items = lambda: []
    state.line_faults = lambda: []
    state.unfit = lambda _: False
    state.gate = lambda _: (False, 'A required cart repair remains unresolved.')
    state.calls_sent = []
    def act(name, args):
        state.calls_sent.append(name)
        state.finished = True
        return {'finished':True, 'observation':{}}
    state.act = act
    state.observe = lambda *_: None
    state.show = json.dumps
    return state


def main():
    ns = extract()
    results = []
    for method in ('direct_order', 'finish'):
        state = cart_fixture()
        ns[method](state)
        assert state.calls_sent == ['buy']
        results.append({'probe':method+'_ignores_negative_gate', 'observed':True,
                        'meaning':'The terminal path sends buy after gate returns False.'})
    gate_state = SimpleNamespace(process=[{'when':'order','step':'Obtain approval','messages':1}],
                    finished=False, roles={'buy':'buy'}, tools=[], request='Obtain approval before ordering.',
                    done_log=[], notes=[], msg_limit=1, msg_used=0)
    gate_state.gate_due = lambda _: True
    gate_state.ledger = gate_state.cart_text = lambda: ''
    gate_state.left = lambda: 5
    gate_state.ask_json = lambda *_: None
    answer = ns['gate'](gate_state, 'order')
    assert answer == (True, '')
    results.append({'probe':'gate_unavailable_response', 'observed':answer,
                    'meaning':'The gate returns permission without proof when its model response is unavailable.'})
    retry = SimpleNamespace(calls=0, waited=0, trace=[], no_credit=False)
    retry.model_room = lambda: 1000
    retry.physical_calls = 0
    def denied(*_):
        retry.physical_calls += 1
        return None, 402
    retry.call_model = denied
    ns['think'](retry, [], ['light'], 100)
    assert (retry.physical_calls, retry.calls, retry.no_credit) == (4, 1, True)
    results.append({'probe':'402_retry_accounting', 'physical_calls':retry.physical_calls,
                    'counted_calls':retry.calls, 'nominal_wait_seconds':retry.waited,
                    'meaning':'A persistent 402 triggers four requests and 34 seconds of scheduled waits before the latch.'})
    transport = SimpleNamespace(cost=.08, trace=[], model_room=lambda: 1000)
    client = SimpleNamespace(post_verbose=lambda *_: SimpleNamespace(
        data={'choices':[{'message':{'content':'ok'}}], 'usage':{}}, error=None))
    ns['_llm_within'] = lambda _: client
    ns['call_model'](transport, 'synthetic/model', [], 100, None)
    assert transport.cost == .08
    results.append({'probe':'missing_cost', 'before':.08, 'after':transport.cost,
                    'meaning':'A successful response without usage.cost consumes zero recorded dollars.'})
    value = ns['scaled_same'](None, '1000 mAh', '10 V')
    assert value is True
    results.append({'probe':'dimensionally_unrelated_figures', 'required':'1000 mAh', 'page':'10 V',
                    'observed':value,
                    'meaning':'The reference-mismatch helper accepts matching stripped digits across unrelated units; it does not prove physical equivalence.'})
    output = {'agent_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'offline_only':True, 'paid_calls':0, 'probes':results,
              'limitation':'Synthetic dependency stubs isolate the shown control-flow branches. These are not live tasks, full-agent tests, or proof that a branch caused a particular race failure.'}
    (ROOT/'offline-contract-probes.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
