"""Reconcile released race metadata and episode feedback without running agents."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    value = json.loads(path.read_text())
    return value.get('data', value)


def main():
    runs, records, races = [], [], []
    for directory in sorted((ROOT/'public').iterdir()):
        if not (directory/'runs.json').exists():
            continue
        index = {r['eval_run_id']: r for r in read(directory/'runs.json')}
        for path in sorted((directory/'details').glob('*.json')):
            detail = read(path)
            run = index[detail['eval_run_id']]
            items = detail['items']['items']
            assert detail['items']['detail_released']
            passed = sum(float(i['paid_reward']) for i in items)
            summary = detail['score_components_summary']
            assert len(items) == summary['task_count']
            assert abs(passed/len(items)-summary['mean_aggregate_reward']) < 1e-9
            base = {'agent': directory.name, 'phase': run['phase'], 'race': run.get('race_number'),
                    'run_id': run['eval_run_id'], 'validator': run['validator_hotkey'],
                    'world': detail['environment']['catalog_epoch'], 'tasks': len(items), 'passed': passed,
                    'cost_usd': detail['sandbox_metadata']['inference_cost_usd'],
                    'duration_seconds': detail['sandbox_metadata']['duration_seconds'],
                    'inference_total': detail['inference_total'], 'inference_errors': detail['inference_failure_count']}
            categories, terminations, issues = Counter(), Counter(), Counter()
            for item in items:
                feedback_path = directory/f"feedback/{item['episode_result_id']}.json"
                feedback = read(feedback_path) if feedback_path.exists() else None
                if feedback:
                    assert item['item_id'] == feedback['task_id']
                    assert item['paid_reward'] == feedback['paid_reward']
                    categories[feedback['failure_category'] or 'passed'] += 1
                    terminations[feedback['termination_code']] += 1
                    for issue in feedback.get('observable_issues') or []:
                        issues[json.dumps(issue, sort_keys=True)] += 1
                records.append({k: base[k] for k in ['agent','phase','race','run_id','validator','world']} |
                               {'task': item['item_id'], 'reward': item['paid_reward'], 'steps': item['step_count'],
                                'episode': item['episode_result_id'],
                                'failure': feedback['failure_category'] if feedback else None,
                                'termination': feedback['termination_code'] if feedback else None})
            runs.append(base | {'failure_categories': dict(categories), 'terminations': dict(terminations),
                                'observable_issues': dict(issues)})
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run['agent'], run['phase'], run['race'])].append(run)
    aggregate = []
    for (agent, phase, race), group in grouped.items():
        aggregate.append({'agent': agent, 'phase': phase, 'race': race, 'runs': len(group),
                          'tasks': sum(r['tasks'] for r in group), 'passed': sum(r['passed'] for r in group),
                          'mean_cost_usd': statistics.mean(r['cost_usd'] for r in group),
                          'min_cost_usd': min(r['cost_usd'] for r in group),
                          'max_cost_usd': max(r['cost_usd'] for r in group),
                          'mean_duration_seconds': statistics.mean(r['duration_seconds'] for r in group),
                          'inference_errors': sum(r['inference_errors'] for r in group),
                          'failure_categories': dict(sum((Counter(r['failure_categories']) for r in group), Counter())),
                          'terminations': dict(sum((Counter(r['terminations']) for r in group), Counter()))})
    pairs = []
    for race in (22, 23):
        for other in ('tosky_submitted_v1', 'pooh', 'chidori', 'giyu'):
            left = {(r['world'], r['task'], r['validator']): r for r in records if r['agent'] == 'pirate' and r['race'] == race}
            right = {(r['world'], r['task'], r['validator']): r for r in records if r['agent'] == other and r['race'] == race}
            common = left.keys() & right.keys()
            cells = defaultdict(list)
            for key in sorted(common):
                a, b = left[key], right[key]
                name = ('both_pass' if b['reward'] else 'pirate_only') if a['reward'] else ('other_only' if b['reward'] else 'both_fail')
                cells[name].append({'task': a['task'], 'world': a['world'], 'validator': a['validator'],
                                    'pirate_episode': a['episode'], 'other_episode': b['episode'],
                                    'pirate_failure': a['failure'], 'other_failure': b['failure']})
            pairs.append({'race': race, 'other': other, 'matched_task_validator_pairs': len(common),
                          'counts': {k: len(v) for k,v in cells.items()}, 'episodes': dict(cells)})
    for path in sorted((ROOT/'public/races').glob('*.json')):
        data = read(path)
        number = data['race']['race_number']
        leaders = sorted((q for q in data['qualifiers'] if q.get('race_score') is not None),
                         key=lambda q: (-q['race_score'], q['agent_name']))
        rows = []
        for q in data['qualifiers']:
            current = next((w for w in q.get('window', []) if w['race_number'] == number), {})
            rows.append({k:q.get(k) for k in ['agent_version_id','agent_name','version_number','race_score','weighted_score','race_rank','is_discarded']} |
                        {'raw_rank': current.get('rank'), 'seed': current.get('is_seed')})
        races.append({'race': number, 'metadata': data['race'], 'participant_rows': len(data['qualifiers']),
                      'raw_top': [{k:q.get(k) for k in ['agent_name','version_number','race_score']} for q in leaders[:5]], 'agents': rows})
    output = {'runs': runs, 'aggregate': aggregate, 'paired_results': pairs, 'races': races,
              'note': 'Raw task outcomes are matched only on identical world, task and validator. Replicate episodes are not independent tasks; feedback categories are the single published category, not an exhaustive diagnosis.'}
    (ROOT/'public-analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    (ROOT/'public-task-results.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps({'runs':len(runs),'episodes':len(records),'aggregate':aggregate,
                      'pairs': [{k:v for k,v in p.items() if k!='episodes'} for p in pairs]},indent=2))


if __name__ == '__main__':
    main()
