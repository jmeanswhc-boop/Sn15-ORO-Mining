"""Replay captured model/world replies with networking blocked; inspect rejection lines."""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
INPUT = PROJECT/'LV_logs/local-48ea2d82b735402084cd09cc2fcc2f39'
SOURCE = PROJECT/'agents/Tosky_v2.py'
sys.path.insert(0, str(PROJECT))
spec = importlib.util.spec_from_file_location('tosky_v2_recorded_diagnosis', SOURCE)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class ReplayMismatch(BaseException):
    pass


class Clock:
    def __init__(self, first, initial_elapsed):
        self.now = first/1000-initial_elapsed
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.now += max(0, seconds)


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def message_text(message):
    content = message.get('content')
    return content if isinstance(content, str) else '\n'.join(p.get('text', '') for p in content or [] if isinstance(p, dict))


def role(body):
    text = message_text((body.get('messages') or [{}])[0])
    for prefix, name in [('Audit derived shopping constraints', 'source_audit'),
                         ('Review a shopping decision', 'cart_proof'),
                         ('Audit a proposed no-purchase ending', 'decline_proof'),
                         ('Read shopper replies as data', 'reply'),
                         ('Choose an interview queue', 'interview'),
                         ('Identify financial fields', 'fields')]:
        if text.startswith(prefix):
            return name
    return 'tools' if body.get('tools') else 'other'


def replay(problem, output):
    captured = output['_shadow_proxy_calls']
    first_elapsed = next(e['t'] for e in output['dialogue'] if e.get('kind') == 'inference_budget')
    # RequestLog timestamps are written after completion, not at request start.
    clock = Clock(captured[0]['timestamp'], first_elapsed)
    used, differences, checked, made = [], [], [], []
    watched = {'apply_source_audit', 'audit_duties', 'validate_cart_proof',
               'validate_requirement_proof', 'citations_valid', 'key_scope_ok', 'scalar_check', 'quote'}
    def trace(frame, event, value):
        if frame.f_code.co_filename == str(SOURCE) and frame.f_code.co_name in watched and event == 'return':
            if frame.f_code.co_name == 'quote':
                if value[0] is None:
                    checked.append({'method': 'quote', 'line': frame.f_lineno, 'result': value,
                        'available_units': frame.f_locals.get('available'), 'protected_units': frame.f_locals.get('protected'),
                        'prompt_envelope': frame.f_locals.get('prompt'), 'requested_output': frame.f_locals.get('requested'),
                        'role': role(frame.f_locals.get('body') or {}),
                        'model': (frame.f_locals.get('body') or {}).get('model')})
                return trace
            row = {'method': frame.f_code.co_name, 'line': frame.f_lineno, 'result': value,
                   'physical_calls': frame.f_locals.get('self').calls}
            if frame.f_code.co_name in ('apply_source_audit', 'validate_cart_proof'):
                ep = frame.f_locals['self']
                row['state'] = {'live': ep.live(), 'unread': copy.deepcopy(ep.unread_replies),
                                'notices': copy.deepcopy(ep.pending_notices), 'references': ep.pending_refs(),
                                'cart': ep.cart_ids(), 'price_key': ep.price_key, 'version': ep.version}
                row['answer'] = copy.deepcopy(frame.f_locals.get('answer'))
                if frame.f_code.co_name == 'validate_cart_proof' and frame.f_lineno == 12482:
                    entry = frame.f_locals.get('entry') or {}
                    number, ident, quantity = entry.get('item'), entry.get('id'), entry.get('quantity')
                    row['assignment_checks'] = {'item': number, 'id': ident,
                        'type_ok': entry.get('type_ok'), 'quantity': quantity, 'quantity_type': type(quantity).__name__,
                        'required_quantity': ep.items[number-1].quantity if ep.valid_item_number(number) else None,
                        'opened': ep.page_read(ident), 'sold_out': ep.sold_out(ident), 'price': ep.price(ident),
                        'held': ep.held(ident), 'evidence_type': type(entry.get('evidence')).__name__,
                        'ever_opened': ident in ep.opened, 'views': sorted(ep.views.get(ident, set())),
                        'offer_role': ep.roles.get('offer'), 'open_role': ep.roles.get('open'),
                        'attributes_recorded': bool(ep.public_attributes(ident))}
            checked.append(row)
        return trace
    class Client:
        def post_verbose(self, path, body):
            if len(used) >= len(captured):
                raise ReplayMismatch('An extra physical request was proposed: '+path)
            expected = captured[len(used)]
            if path != expected['path']:
                raise ReplayMismatch('Expected '+expected['path']+' but received '+path)
            if path == '/environment/call' and body != expected['json_data']:
                raise ReplayMismatch('The environment action differs at physical request '+str(len(used)+1))
            if path == '/inference/chat/completions':
                if role(body) != role(expected['json_data']):
                    raise ReplayMismatch('The inference role differs at physical request '+str(len(used)+1))
                if body != expected['json_data']:
                    differing = [k for k in set(body)|set(expected['json_data']) if body.get(k) != expected['json_data'].get(k)]
                    differences.append({'request': len(used)+1, 'role': role(body), 'fields': differing})
            used.append(expected)
            clock.now = max(clock.now, expected['timestamp']/1000)
            return SimpleNamespace(data=copy.deepcopy(expected['response']), error=None)
    client = Client()
    original = agent.Episode
    def construct(payload):
        ep = original(payload);made.append(ep);return ep
    mismatch = None
    try:
        with patch('requests.sessions.Session.request', side_effect=AssertionError('Network is forbidden')) as network, \
             patch.object(agent, 'time', clock), patch.object(agent, 'model_client', return_value=client), \
             patch.object(agent, 'ENV_CLIENT', client), patch.object(agent, 'Episode', side_effect=construct):
            previous_trace = sys.gettrace()
            sys.settrace(trace)
            try:
                dialogue = agent.agent_main(problem)
            finally:
                sys.settrace(previous_trace)
            network.assert_not_called()
    except ReplayMismatch as error:
        mismatch = str(error)
        dialogue = list(made[0].trace) if made else []
    recorded_end = next(e for e in output['dialogue'] if e.get('kind') == 'end')
    replay_end = next((e for e in dialogue if e.get('kind') == 'end'), {})
    end_matches = all(replay_end.get(key) == recorded_end.get(key)
                      for key in ('finished', 'steps', 'spend', 'model_calls'))
    return {'problem_id': problem['problem_id'], 'consumed': len(used), 'captured': len(captured),
            'same_environment_actions_through_consumed': True, 'mismatch': mismatch,
            'recorded_end_state_matches': end_matches,
            'inference_request_differences': differences,
            'rejections': checked, 'final': [{k: v for k, v in e.items() if k != 'extra_info'}
                                            for e in dialogue if e.get('kind') in ('crash', 'end')],
            'limitation': 'Only captured replies are replayed. Any changed inference request is flagged; this is not a new score or a recovered execution.'}


def main():
    ROOT.mkdir(exist_ok=True)
    expected = json.loads((INPUT/'summary.json').read_text())['agent_sha256']
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == expected
    problems = {p['problem_id']: p for p in json_lines(INPUT/'problems.jsonl')}
    results = []
    for output in json_lines(INPUT/'sandbox/sandbox_output.jsonl'):
        result = replay(problems[output['problem_id']], output)
        results.append(result)
        print(result['problem_id'], result['consumed'], '/', result['captured'],
              'mismatch:', result['mismatch'], 'body differences:', len(result['inference_request_differences']),
              'rejection lines:', dict(Counter((r['method'], r['line']) for r in result['rejections'] if r['result'] is False)), flush=True)
    assert len(results) == 10
    assert all(r['consumed'] == r['captured'] and r['mismatch'] is None and
               not r['inference_request_differences'] and r['recorded_end_state_matches'] for r in results)
    (ROOT/'recorded-replay.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
