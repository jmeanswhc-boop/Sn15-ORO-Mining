"""Join the ten local cases, actual model responses and traced rejection points."""
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics

from replay_recorded import ROOT, PROJECT, INPUT, SOURCE, agent, json_lines, message_text, role


def packet(body):
    text = message_text((body.get('messages') or [{}])[0])
    position = text.rfind('\n{')
    return json.loads(text[position+1:]) if position >= 0 else None


def main():
    outputs = json_lines(INPUT/'sandbox/sandbox_output.jsonl')
    episodes = {e['episode']['session_id']: e['episode'] for e in json_lines(INPUT/'episode_results.jsonl')}
    problems = {p['problem_id']: p for p in json_lines(INPUT/'problems.jsonl')}
    replay = {r['problem_id']: r for r in json.loads((ROOT/'recorded-replay.json').read_text())}
    expected = json.loads((INPUT/'summary.json').read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == expected['agent_sha256']
    totals = Counter();by_role = defaultdict(lambda: {'requests': 0, 'cost_usd': Decimal(0), 'latency_seconds': 0.0})
    cases, decisions = [], []
    for output in outputs:
        session = output['problem_id'];episode = episodes[session]
        assert episode['task_id'] in expected['task_roster'] and episode['pack_sha256'] == expected['pack_sha256']
        assert output['status'].casefold() in ('success', 'completed'), output['status']
        events = [{k: v for k, v in event.items() if k != 'extra_info'} for event in output['dialogue']]
        brief = next(event['brief'] for event in events if event.get('kind') == 'brief')
        terminal_names = {brief['roles'].get('buy'), brief['roles'].get('refuse')}-{None}
        first_proof = None;inference = 0;decision_rows = []
        for index, call in enumerate(output['_shadow_proxy_calls'], 1):
            if call['path'] == '/environment/call':
                if call['json_data']['action']['name'] in terminal_names:
                    totals['terminal_environment_calls'] += 1
                continue
            if call['path'] != '/inference/chat/completions':
                continue
            inference += 1
            body, response = call['json_data'], call['response'];name = role(body)
            cost = Decimal(str((response.get('usage') or {}).get('cost') or 0))
            by_role[name]['requests'] += 1;by_role[name]['cost_usd'] += cost
            by_role[name]['latency_seconds'] += call['duration_ms']/1000
            if name not in ('source_audit', 'cart_proof'):
                continue
            choice = (response.get('choices') or [{}])[0]
            raw = message_text(choice.get('message') or {})
            answer = agent.obj_of(raw)
            row = {'session_id': session, 'task_id': episode['task_id'], 'proxy_request': index,
                   'inference_request': inference, 'role': name, 'model': body['model'],
                   'finish_reason': choice.get('finish_reason'), 'parsed': answer is not None,
                   'packet': packet(body), 'answer': answer, 'cost_usd': str(cost)}
            decisions.append(row);decision_rows.append(row)
            totals[name] += 1
            if name == 'cart_proof' and first_proof is None:first_proof = row
        traced = replay[session]
        proof_rejections = [r for r in traced['rejections'] if r['method'] == 'validate_cart_proof']
        assignment_rejections = []
        for rejection in proof_rejections:
            checks = rejection.get('assignment_checks')
            if checks:
                reason = ('reported_wrong_product_type' if checks['type_ok'] is not True else
                          'required_view_missing' if not checks['opened'] else
                          'free_text_evidence' if checks['evidence_type'] != 'list' else 'citation_mismatch')
                assignment_rejections.append({'inference_request': rejection['physical_calls'], 'reason': reason, 'checks': checks})
        denials = [e for e in events if e.get('kind') == 'request_budget_denied']
        case = {'task_id': episode['task_id'], 'session_id': session,
                'subject': brief['items'][0]['name'], 'request': episode['request_text'],
                'score': episode['verdict']['score'], 'terminal_reason': episode['terminal_reason'],
                'steps': episode['step_count'], 'max_steps': problems[session]['environment']['policy_view']['max_steps'],
                'seconds': output['execution_time'], 'inference_requests': output['inference_total'],
                'inference_failures': output['inference_failure_count'],
                'cost_usd': output['_shadow_inference_usage']['inference_cost_usd'],
                'source_audit_attempts': sum(d['role'] == 'source_audit' for d in decision_rows),
                'source_audit_accepted': sum(e.get('kind') == 'source_audit' for e in events),
                'review_rejection_lines': dict(Counter(str(r['line']) for r in proof_rejections)),
                'assignment_rejections': assignment_rejections,
                'first_review': {key: first_proof['answer'].get(key) for key in ('complete', 'source_complete', 'problems')} if first_proof and first_proof['answer'] else None,
                'local_budget_denials': len([e for e in events if e.get('kind') == 'budget_denied']),
                'last_remaining_at_local_denial': denials[-1].get('remaining_usd') if denials else None,
                'crashes': [e for e in events if e.get('kind') == 'crash'],
                'replay': {'consumed': traced['consumed'], 'captured': traced['captured'], 'mismatch': traced['mismatch'],
                           'request_parameter_differences': traced['inference_request_differences']}}
        cases.append(case)
    history = []
    for name in ['local-57eb955c7dcc48dc8d62f192a5d8760e', 'local-f6809126cbb043e6a3b152a7582d982e',
                 'local-0e47aedee480429ba19d94c1909dd7f3']:
        summary = json.loads((PROJECT/'LV_logs'/name/'summary.json').read_text())
        assert summary['pack_sha256'] == expected['pack_sha256'] and summary['world_id'] == expected['world_id']
        subset = [t for t in summary['tasks'] if t['task_id'] in expected['task_roster']]
        assert len(subset) == 10
        history.append({'run_id': name, 'agent_sha256': summary['agent_sha256'],
                        'same_world_and_pack': True, 'subset_correct': sum(t['score'] for t in subset), 'tasks': subset})
    for case in cases:
        case['prior_control_outcomes'] = {h['run_id']: next(t['score'] for t in h['tasks'] if t['task_id'] == case['task_id']) for h in history}
    audit_answers = [d for d in decisions if d['role'] == 'source_audit']
    proof_answers = [d for d in decisions if d['role'] == 'cart_proof']
    metrics = {'tasks': len(cases), 'correct': sum(c['score'] for c in cases),
               'inference_requests': sum(c['inference_requests'] for c in cases),
               'inference_failures': sum(c['inference_failures'] for c in cases),
               'cost_usd': str(sum(Decimal(str(c['cost_usd'])) for c in cases)),
               'mean_seconds': statistics.mean(c['seconds'] for c in cases),
               'agent_crashes': sum(len(c['crashes']) for c in cases),
               'terminal_environment_calls': totals['terminal_environment_calls'],
               'cases_with_local_budget_denial': sum(c['local_budget_denials'] > 0 for c in cases),
               'source_audits': len(audit_answers), 'accepted_source_audits': sum(c['source_audit_accepted'] for c in cases),
               'audit_unparseable': sum(d['answer'] is None for d in audit_answers),
               'audit_complete_false': sum(isinstance(d['answer'], dict) and d['answer'].get('complete') is False for d in audit_answers),
               'proof_responses': len(proof_answers),
               'proof_finish_reasons': dict(Counter(d['finish_reason'] for d in proof_answers)),
               'proof_rejection_lines': dict(Counter(str(r['line']) for t in replay.values() for r in t['rejections'] if r['method'] == 'validate_cart_proof')),
               'assignment_rejection_reasons': dict(Counter(r['reason'] for c in cases for r in c['assignment_rejections'])),
               'all_task_prompts_and_environment_actions_replayed': all(t['mismatch'] is None and t['consumed'] == t['captured'] for t in replay.values()),
               'fully_identical_request_replays': sum(not t['inference_request_differences'] for t in replay.values()),
               'replayed_physical_requests': sum(t['consumed'] for t in replay.values()),
               'prior_pass_for_every_selected_task': all(any(c['prior_control_outcomes'].values()) for c in cases)}
    roles = {k: {**v, 'cost_usd': str(v['cost_usd'])} for k, v in by_role.items()}
    (ROOT/'analysis.json').write_text(json.dumps({'input_run': expected['run_id'], 'agent_sha256': expected['agent_sha256'],
        'world_id': expected['world_id'], 'pack_sha256': expected['pack_sha256'], 'selection_seed': expected['selection_seed'],
        'metrics': metrics, 'inference_by_role': roles, 'cases': cases, 'historical_controls': history,
        'scope': 'Recorded local practice executions; no new model calls, no new score, no source modification.'}, indent=2)+'\n')
    (ROOT/'recorded-decisions.json').write_text(json.dumps(decisions, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))
    print('Roles:', json.dumps(roles))
    for case in sorted(cases, key=lambda c: c['task_id']):
        print(case['task_id'], case['subject'], case['steps'], case['review_rejection_lines'],
              [r['reason'] for r in case['assignment_rejections']])


if __name__ == '__main__':
    main()
