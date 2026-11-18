"""Reconcile the supplied local runs and save evidence of their observed failures."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
RUNS = ['local-71b9db485e8b423d986df21dacab6c2d', 'local-cb574c03f70841abb0efec38dba1f2f0']


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(run):
    base = PROJECT/'LV_logs'/run
    summary = json.loads((base/'summary.json').read_text())
    assert summary['agent_sha256'] == hashlib.sha256((PROJECT/'agents/Tosky_v3.py').read_bytes()).hexdigest()
    latest = {}
    for row in jsonl(base/'sandbox/inference_stats.jsonl'):
        latest[(row['problem_id'], row['instance_id'], row['execution_id'])] = row
    usage = {row['problem_id']: row for row in latest.values()}
    assert len(usage) == len(latest) == summary['task_count']
    outputs = {row['problem_id']: row for row in jsonl(base/'sandbox/sandbox_output.jsonl')}
    episodes = [row['episode'] for row in jsonl(base/'episode_results.jsonl')]
    result = {'run_id': run, 'source_sha256': summary['agent_sha256'], 'pack_sha256': summary['pack_sha256'],
              'world_id': summary['world_id'], 'selection_seed': summary['selection_seed'],
              'tasks': len(episodes), 'correct': sum(e['verdict']['score'] for e in episodes),
              'cost_usd': sum(u['inference_cost_usd'] for u in usage.values()),
              'physical_model_calls': sum(u['inference_total'] for u in usage.values()),
              'failed_model_calls': sum(u['inference_failed'] for u in usage.values()),
              'missing_costs': sum(u['inference_cost_missing'] for u in usage.values()),
              'mean_task_seconds': statistics.mean(o['execution_time'] for o in outputs.values()),
              'max_task_seconds': max(o['execution_time'] for o in outputs.values()),
              'categories': dict(Counter(e['verdict']['category'] or 'passed' for e in episodes)),
              'task_results': [], 'failures': []}
    for e in episodes:
        u, out = usage[e['session_id']], outputs[e['session_id']]
        assert abs(out['_shadow_inference_usage']['inference_cost_usd'] - u['inference_cost_usd']) < 1e-9
        row = {'task': e['task_id'], 'session': e['session_id'], 'score': e['verdict']['score'],
               'category': e['verdict']['category'], 'request': e['request_text'], 'cost_usd': u['inference_cost_usd'],
               'model_calls': u['inference_total'], 'model_errors': u['inference_failed'],
               'seconds': out['execution_time'], 'world_steps': e['step_count'], 'terminal': e['terminal_reason'],
               'diagnostic_counts': dict(Counter(d.get('kind', d.get('role')) for d in out['dialogue']))}
        result['task_results'].append(row)
        if not row['score']:
            selected = [{k: v for k, v in d.items() if k != 'extra_info'} for d in out['dialogue']
                        if d.get('kind') in ['brief', 'plan', 'message', 'cart_check', 'review_dispute', 'review_boundary',
                          'compact_repair', 'request_budget_denied', 'source_coverage', 'end', 'gate', 'terminal_hold']]
            records = jsonl(base/f"sandbox/request_log_{e['session_id']}.jsonl")
            calls = [d for d in records if d.get('json_data') and d.get('path') == '/inference/chat/completions']
            final = []
            for n, d in list(enumerate(calls, 1))[-8:]:
                data = d.get('response') or {}
                if isinstance(data, str):
                    data = json.loads(data)
                message = (data.get('choices') or [{}])[0].get('message') or {}
                final.append({'number': n, 'model': d['json_data']['model'], 'content': message.get('content'),
                              'tool_calls': message.get('tool_calls'), 'cost': (data.get('usage') or {}).get('cost')})
            actions = [d for d in e['ledger']['episode'] if d.get('kind') in ['action', 'observation', 'finish', 'basket']]
            evidence = {**row, 'diagnostics': selected, 'final_model_outputs': final, 'world_actions_and_observations': actions}
            folder = ROOT/'local-failures'/run
            folder.mkdir(parents=True, exist_ok=True)
            (folder/(e['task_id']+'.json')).write_text(json.dumps(evidence, indent=2) + '\n')
            result['failures'].append({k: row[k] for k in ['task','category','request','cost_usd','model_calls','seconds','world_steps','terminal','diagnostic_counts']})
    assert abs(result['correct']/result['tasks'] - summary['aggregate_score']) < 1e-10
    return result


def main():
    runs = [analyze(run) for run in RUNS]
    left, right = [{r['task']: r for r in run['task_results']} for run in runs]
    common = sorted(left.keys() & right.keys())
    assert all(left[k]['request'] == right[k]['request'] for k in common)
    pairing = {'shared_tasks': len(common), 'identical_shared_requests': True,
               'both_pass': [k for k in common if left[k]['score'] and right[k]['score']],
               'both_fail': [k for k in common if not left[k]['score'] and not right[k]['score']],
               'pass_to_fail': [k for k in common if left[k]['score'] and not right[k]['score']],
               'fail_to_pass': [k for k in common if not left[k]['score'] and right[k]['score']],
               'new_tasks': sorted(right.keys()-left.keys()),
               'new_tasks_passed': sum(right[k]['score'] for k in right.keys()-left.keys())}
    output = {'runs': runs, 'paired_comparison': pairing,
              'note': 'Same source and public practice pack; this is run-to-run variation, not evidence of a source-update regression. Local practice scores are not public qualifying or hidden race scores.'}
    (ROOT/'local-analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'runs': [{k:v for k,v in r.items() if k not in ['task_results','failures']} for r in runs], 'paired': pairing}, indent=2))
    for r in runs:
        for f in r['failures']:
            print(json.dumps({'run': r['run_id'], **{k: v for k, v in f.items() if k != 'diagnostic_counts'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
