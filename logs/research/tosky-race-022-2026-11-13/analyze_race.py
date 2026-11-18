"""Validate and summarize released evidence without importing any agent code."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
AGENTS = ('tosky', 'chidori', 'pooh')


def load(path):
    value = json.loads(path.read_text())
    return value.get('data', value)


def mean(values):
    return statistics.mean(values) if values else None


def main():
    records = []
    tasks = defaultdict(lambda: defaultdict(list))
    runs = {}
    missing = []
    inconsistencies = []
    for path in sorted((ROOT/'race-details').glob('*/*.json')):
        agent = path.parent.name
        data = load(path)
        index = data['items']
        assert index['detail_released'] is True, path
        assert index['expected_item_count'] == index['persisted_item_count'] == len(index['items']) == 90, path
        run = data['eval_run_id']
        run_record = {
            'agent': agent, 'run': run, 'validator': data['validator_hotkey'],
            'included': data['is_included'], 'status': data['status'],
            'correct': sum(item['paid_reward'] for item in index['items']),
            'items': len(index['items']), 'usd': data['sandbox_metadata']['inference_cost_usd'],
            'sandbox_seconds': data['sandbox_metadata']['duration_seconds'],
            'inferences': data['inference_total'], 'inference_errors': data['inference_failure_count'],
            'feedback_available': 0,
        }
        runs[run] = run_record
        for item in index['items']:
            episode = item['episode_result_id']
            row = {'agent': agent, 'run': run, 'validator': data['validator_hotkey'],
                   'task': item['item_id'], 'episode': episode, 'passed': bool(item['paid_reward']),
                   'steps': item['step_count'], 'feedback_available': False}
            feedback = ROOT/'race-feedback'/agent/(episode+'.json')
            if feedback.exists():
                wrapper = json.loads(feedback.read_text())
                d = wrapper.get('data') or {}
                valid = not wrapper.get('error') and not wrapper.get('stale') and d.get('diagnostics_state') == 'available'
                if valid:
                    for field, value in [('task_id', item['item_id']), ('episode_result_id', episode), ('paid_reward', item['paid_reward'])]:
                        if d.get(field) != value:
                            inconsistencies.append({'episode': episode, 'field': field, 'feedback': d.get(field), 'index': value})
                    row.update({'feedback_available': True, 'failure_category': d.get('failure_category'),
                                'termination_code': d.get('termination_code'), 'end_reason': d.get('end_reason'),
                                'issues': d.get('observable_issues'), 'source': str(feedback.relative_to(ROOT)),
                                'questions': [{'step': turn['step_index'], 'question': turn['message'], 'reply': turn.get('shopper_message')}
                                              for turn in d.get('interaction', []) if turn.get('message')],
                                'tools': dict(Counter(call['tool_name'] for turn in d.get('interaction', []) for call in turn.get('calls', []))),
                                'tool_error_codes': dict(Counter(str(call['error_code']) for turn in d.get('interaction', [])
                                                                for call in turn.get('calls', []) if call.get('error_code')))})
                    run_record['feedback_available'] += 1
            if not row['feedback_available']:
                missing.append({'agent': agent, 'run': run, 'episode': episode})
            records.append(row)
            tasks[row['task']][agent].append(row)
    assert len(runs) == 6 and len(records) == 540 and len(tasks) == 90
    assert all(len(agents[a]) == 2 for agents in tasks.values() for a in AGENTS)
    task_rows = [{'task': task, 'wins': {a: sum(r['passed'] for r in agents[a]) for a in AGENTS}, 'records': dict(agents)}
                 for task, agents in sorted(tasks.items())]
    result = {'analyzed_at': datetime.now(timezone.utc).isoformat(), 'complete': not missing and not inconsistencies,
              'coverage': {'expected_feedback': 540, 'available_feedback': len(records)-len(missing),
                           'missing': missing, 'inconsistencies': inconsistencies},
              'runs': list(runs.values()), 'agents': {}, 'pairwise': {}, 'task_comparison': task_rows,
              'shared_failures_all_six': [t['task'] for t in task_rows if not any(t['wins'].values())]}
    for agent in AGENTS:
        own = [r for r in records if r['agent'] == agent]
        available = [r for r in own if r['feedback_available']]
        own_runs = [r for r in runs.values() if r['agent'] == agent]
        result['agents'][agent] = {
            'correct': sum(r['passed'] for r in own), 'executions': len(own),
            'mean_usd': mean([r['usd'] for r in own_runs]),
            'mean_sandbox_seconds': mean([r['sandbox_seconds'] for r in own_runs]),
            'total_inferences': sum(r['inferences'] for r in own_runs),
            'inference_errors': sum(r['inference_errors'] for r in own_runs),
            'cost_per_successful_execution': sum(r['usd'] for r in own_runs)/sum(r['passed'] for r in own),
            'feedback_available': len(available),
            'categories': dict(Counter(r['failure_category'] or 'passed' for r in available)),
            'terminations': dict(Counter(r['termination_code'] for r in available)),
            'end_reasons': dict(Counter(r['end_reason'] for r in available)),
            'mean_steps': mean([r['steps'] for r in own]),
            'mean_steps_on_pass': mean([r['steps'] for r in own if r['passed']]),
            'mean_steps_on_failure': mean([r['steps'] for r in own if not r['passed']]),
        }
    for other in ('chidori', 'pooh'):
        comparisons = []
        for task, agents in sorted(tasks.items()):
            left = {r['validator']: r for r in agents['tosky']}
            right = {r['validator']: r for r in agents[other]}
            for validator in left.keys() & right.keys():
                comparisons.append({'task': task, 'validator': validator, 'tosky': left[validator], other: right[validator]})
        result['pairwise']['tosky_vs_'+other] = {
            'matched_executions': len(comparisons),
            'both_pass': sum(c['tosky']['passed'] and c[other]['passed'] for c in comparisons),
            'both_fail': sum(not c['tosky']['passed'] and not c[other]['passed'] for c in comparisons),
            'tosky_only': [c for c in comparisons if c['tosky']['passed'] and not c[other]['passed']],
            'other_only': [c for c in comparisons if c[other]['passed'] and not c['tosky']['passed']],
        }
    (ROOT/'race-analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['complete', 'agents']}, indent=2))
    print('Shared all-six failures:', len(result['shared_failures_all_six']))
    print('Coverage:', len(records)-len(missing), '/', len(records), 'inconsistencies:', len(inconsistencies))


if __name__ == '__main__':
    main()
