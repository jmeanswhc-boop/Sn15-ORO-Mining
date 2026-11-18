"""Final consistency and source-preservation checks; strictly offline."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
analysis = json.loads((HERE / 'analysis.json').read_text())
originals = json.loads((ROOT / 'logs/research/tosky-v4-build-2026-11-14/original-hashes.json').read_text())
originals['agents/Tosky_v4.py'] = analysis['current']['source_sha256']
hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in originals}
assert originals == hashes
helper = runpy.run_path(str(ROOT / 'agents/tests/test_tosky_v4.py'))
ns = helper['load']()
audits = []
for run in [analysis['current'], analysis['full']]:
    counts = Counter()
    finish = Counter()
    invalid = []
    need_entries = []
    denials = []
    ratios = []
    messages = []
    max_cost = 0
    for file in sorted((HERE / 'tasks' / run['run']).glob('*.json')):
        data = json.loads(file.read_text())
        counts.update(x.get('kind') for x in data['diagnostics'])
        calls = {c['number']: c for c in data['calls']}
        max_cost = max(max_cost, data['cost'])
        for c in calls.values():
            finish[c['finish_reason']] += 1
            if c['purpose'] != 'GUIDE' and ns['_json_obj'](c['content']) is None:
                invalid.append((data['task'], c['number']))
            if c['purpose'] == 'HEAR':
                obj = ns['_json_obj'](c['content']) or {}
                need_entries.extend({'task': data['task'], 'entry': e} for e in obj.get('needs', []))
        for i, d in enumerate(data['diagnostics']):
            if d.get('kind') == 'budget_denied':
                denials.append({'task': data['task'], 'score': data['score'], 'model': d['model'], 'spent': d['spent']})
            if d.get('kind') == 'ask':
                messages.append({'task': data['task'], 'question': d['q'].endswith('?')})
            if d.get('kind') == 'inference_budget':
                assert not d['breached'] and d['uncertain_usd'] < 1e-9
                ratios.append(d['prompt_estimate'] / calls[d['attempt']]['usage']['prompt_tokens'])
    assert not invalid
    assert not counts['soft_crash']
    assert finish['length'] == 0
    assert counts['inference_budget'] == run['calls']
    audits.append({'run': run['run'], 'source_verified': True, 'max_task_cost': max_cost,
                   'diagnostic_counts': dict(counts), 'finish_reasons': dict(finish),
                   'nonplanner_json_invalid': invalid, 'local_denials': denials,
                   'new_interview_requirements': need_entries, 'shopper_messages': len(messages),
                   'preference_questions': sum(x['question'] for x in messages),
                   'median_input_estimate_ratio': statistics.median(ratios)})

probes = json.loads((HERE / 'probes.json').read_text())
assert probes['watch_question']['reproduced_choice'] == 'options'
assert probes['full_hoodie_question']['reproduced_choice'] == 'season'
assert probes['full_clock_recency_loss']['position_in_newest_first_candidates'] == 42
assert probes['full_category_alternative_loss']['jeans_owner'] == 0
assert analysis['full']['correct'] == 21 and analysis['current']['correct'] == 14
assert len(analysis['repeat_comparison']['both_fail']) == 4
report = (HERE / 'report.txt').read_text()
assert 'at 496 cents' in report
output = {'passed': True, 'source_hashes_unchanged': hashes, 'runs': audits,
          'probes_verified': True, 'scope': 'Offline analysis; no agent changes or new benchmark execution.',
          'artifacts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [HERE / 'analysis.json', HERE / 'probes.json', HERE / 'report.txt']}}
(HERE / 'verification.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({'passed': True, 'source_hashes_unchanged': hashes,
                  'runs': [{k:v for k,v in r.items() if k not in ['diagnostic_counts','local_denials','new_interview_requirements']} for r in audits]}, indent=2))
