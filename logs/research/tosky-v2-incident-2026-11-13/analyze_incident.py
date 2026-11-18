"""Analyze released receipts without inference or modifying any agent."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PREVIOUS = PROJECT/'logs/research/tosky-race-022-2026-11-13'


def read(path):
    return json.loads(path.read_text())


def calls(feedback):
    return [call for turn in feedback['interaction'] for call in turn['calls']]


def basket(feedback):
    cart = {}
    for call in calls(feedback):
        observation = call['observation']
        if not isinstance(observation, dict):
            continue
        if call['tool_name'] == 'put_quantity' and isinstance(observation.get('id'), str) and isinstance(observation.get('qty'), int):
            if observation['qty']:
                cart[observation['id']] = observation['qty']
            else:
                cart.pop(observation['id'], None)
        elif call['tool_name'] == 'trolley_contents' and isinstance(observation.get('lines_lines'), list):
            cart = {line['id']: line['qty'] for line in observation['lines_lines'] if line.get('qty') and isinstance(line.get('id'), str)}
    return cart


def main():
    monitor = read(ROOT/'monitor.json')
    version = next(e for e in monitor['entries'] if e['agent_version_id'] == '06af6d97-a794-7a4c-8000-98022e3cf871')
    baseline = next(e for e in monitor['entries'] if e['agent_version_id'] == '06af5d7e-bfd4-761c-8000-575e592cb36b')
    previous = [read(p)['data'] for p in (PREVIOUS/'qualifying-feedback').glob('*.json')]
    current = []
    for filename in ('run-one.json', 'run-two.json'):
        run = read(ROOT/filename)['data']
        assert run['items']['detail_released'] and run['items']['persisted_item_count'] == 30
        for item in run['items']['items']:
            feedback = read(ROOT/'feedback'/(item['episode_result_id']+'.json'))
            assert not feedback['error'] and not feedback['stale']
            feedback = feedback['data']
            assert feedback['episode_result_id'] == item['episode_result_id']
            assert feedback['task_id'] == item['item_id'] and feedback['paid_reward'] == item['paid_reward']
            current.append(feedback)
    assert len(previous) == len(current) == 60
    source = PROJECT/'agents/Tosky_v2.py'
    original_verification = read(PROJECT/'logs/research/tosky-v2-2026-11-13/verification.json')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == original_verification['sha256']
    for entry in original_verification['preserved_versions']:
        assert hashlib.sha256((PROJECT/entry['path']).read_bytes()).hexdigest() == entry['sha256']
    def metrics(entry, records):
        runs = [run for run in entry['runs'] if run['phase'] == 'QUALIFYING' and run['is_included']]
        cost = sum(Decimal(str(run['sandbox_metadata']['inference_cost_usd'])) for run in runs)
        passed = sum(d['passed'] for d in records)
        return {'correct': passed, 'executions': len(records),
                'mean_run_cost_usd': str(cost/len(runs)),
                'mean_run_seconds': sum(run['sandbox_metadata']['duration_seconds'] for run in runs)/len(runs),
                'inference_requests': sum(run['inference_total'] for run in runs),
                'inference_errors': sum(run['inference_failure_count'] for run in runs),
                'cost_per_correct_usd': str(cost/passed) if passed else None,
                'end_reasons': dict(Counter(d['end_reason'] for d in records)),
                'failure_categories': dict(Counter(d.get('failure_category') or 'passed' for d in records))}
    task_rows = []
    stopped = [d for d in current if d['end_reason'] == 'agent_stopped']
    same_baskets = []
    for d in current:
        matching = [p for p in previous if p['task_id'] == d['task_id'] and p['passed'] and basket(p) == basket(d) and basket(d)]
        if matching and not d['passed']:
            same_baskets.append({'task': d['task_id'], 'episode': d['episode_result_id'], 'basket': basket(d),
                                 'previous_passing_episodes': [p['episode_result_id'] for p in matching]})
    for task in sorted({d['task_id'] for d in current}):
        before = [d for d in previous if d['task_id'] == task]
        after = [d for d in current if d['task_id'] == task]
        task_rows.append({'task': task, 'v4_correct': sum(d['passed'] for d in before),
                          'v2_correct': sum(d['passed'] for d in after),
                          'v2_episodes': [d['episode_result_id'] for d in after]})
    result = {'submitted_version': version['agent_version_id'], 'local_candidate_sha256': source_hash,
              'local_candidate_matches_last_verified_file': True,
              'uploaded_source_bytes_independently_confirmed': False,
              'baseline': metrics(baseline, previous), 'candidate': metrics(version, current),
              'all_sixty_identities_and_rewards_verified': True,
              'tasks_previously_two_of_two_now_zero_of_two': sum(t['v4_correct'] == 2 and t['v2_correct'] == 0 for t in task_rows),
              'stopped_with_nonempty_observed_basket': sum(bool(basket(d)) for d in stopped),
              'stopped_before_38_step_limit': sum(d['step_count'] < 38 for d in stopped),
              'terminal_world_calls': dict(Counter(c['tool_name'] for d in current for c in calls(d)
                                                  if c['tool_name'] in ('submit_purchase', 'decline'))),
              'same_basket_as_a_previous_pass': same_baskets,
              'same_basket_limitation': 'Same products and quantities do not prove identical hidden preferences or complete correctness.',
              'active_qualifier_snapshot': {'agent_version_id': baseline['agent_version_id'],
                                             'is_active_qualifier': baseline['is_active_qualifier'],
                                             'v2_is_active': version['is_active_qualifier']},
              'tasks': task_rows}
    (ROOT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('tasks', 'same_basket_as_a_previous_pass')}, indent=2))
    print('Stopped with an old passing basket:', len(same_baskets))


if __name__ == '__main__':
    main()
