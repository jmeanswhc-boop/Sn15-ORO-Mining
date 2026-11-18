"""Summarize already observed public traces and offline method contracts."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from probe import snapshot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
analysis = json.loads((HERE / 'analysis.json').read_text())
out = {'runs': {}}
for label, run in analysis['runs'].items():
    counts = Counter()
    grouped, needs, lengths, denials = [], [], [], []
    for task in run['tasks']:
        d = json.loads((HERE / 'tasks' / run['run'] / (task['task'] + '.json')).read_text())
        counts.update(x['kind'] for x in d['diagnostics'] if 'kind' in x)
        grouped.extend({'task': task['task'], **x} for x in d['diagnostics'] if x.get('kind') == 'question_coverage')
        denied = sum(x.get('kind') == 'budget_denied' for x in d['diagnostics'])
        if denied:
            denials.append({'task': task['task'], 'score': task['score'], 'denials': denied})
        for c in d['calls']:
            if c['finish_reason'] == 'length':
                lengths.append({'task': task['task'], **{k: c[k] for k in ('purpose', 'model', 'finish_reason', 'usage', 'content')}})
            if c['purpose'] != 'HEAR':
                continue
            try:
                parsed = json.loads(c['content'].strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
            except ValueError:
                continue
            if parsed.get('needs'):
                needs.append({'task': task['task'], 'needs': parsed['needs']})
    out['runs'][label] = {'diagnostic_counts': dict(counts), 'grouped_questions': len(grouped),
                         'grouped_with_new_needs': sum(bool(g.get('new')) for g in grouped),
                         'new_needs': needs, 'length_limited_responses': lengths, 'budget_denials': denials}

out['repeated_search_results'] = {}
for task in ('t-9ab5c575f5', 't-b516252b9f'):
    d = json.loads((HERE / 'tasks' / analysis['runs']['v5']['run'] / (task + '.json')).read_text())
    actions, seen, searches = {}, {}, []
    for event in d['world']:
        if event['kind'] == 'action' and event.get('tool') == 'find_products':
            actions[event['step']] = event['args']
        if event['kind'] == 'observation' and event['step'] in actions:
            ids = tuple(dict.fromkeys(re.findall(r'p-[0-9a-f]{12}', json.dumps(event['observation']))))
            same = [step for step, prior in seen.items() if prior == ids]
            searches.append({'step': event['step'], 'args': actions[event['step']], 'ids': ids, 'same_ordered_ids_as': same})
            seen[event['step']] = ids
    out['repeated_search_results'][task] = searches

out['repeated_negative_review'] = {}
for source in ('Tosky_v4.py', 'Tosky_v5.py'):
    ns, engine, _, _ = snapshot('t-d4e2bc9e66', source, stop=13)
    engine.checked = set()
    engine.verdicts = {}
    engine.judges = lambda limit: [ns['LIGHT']]
    engine.ask_json = lambda *args, **kwargs: {'ok': False, 'problems': ['Synthetic unresolved requirement for a cache-contract check'], 'better': []}
    ids = ['p-1f4aa8507466']
    first, second = engine.check_cart(ids), engine.check_cart(ids)
    verdict = engine.verdicts[engine.check_key(ids)]
    assert first and second == '' and verdict is False
    out['repeated_negative_review'][source] = {'first_returns_hold': bool(first), 'second_returns_hold': bool(second),
                                              'stored_verdict': verdict,
                                              'limit': 'Scripted method contract only; not proof of the exact public grader failure.'}

def method(path, name):
    return ast.dump(next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)

out['unchanged_v4_v5_methods'] = {name: method(ROOT / 'agents/Tosky_v4.py', name) == method(ROOT / 'agents/Tosky_v5.py', name)
                                    for name in ('fuller_label', 'line_faults', 'check_cart', 'finish', 'trim', 'quote_inference')}

hashes = json.loads((ROOT / 'logs/research/tosky-v5-build-2026-11-14/original-hashes.json').read_text())
hashes['agents/Tosky_v5.py'] = analysis['runs']['v5']['source_sha256']
out['preservation'] = {}
for file, expected in hashes.items():
    actual = hashlib.sha256((ROOT / file).read_bytes()).hexdigest()
    assert actual == expected, file
    out['preservation'][file] = actual
out['limits'] = 'No paid inference or environment replay. Original agents/tests unchanged. Method probes use a transport-blocked loader and observed public records. No grading internals used.'
(HERE / 'supplement.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'hashes_match': True, 'unchanged_methods': out['unchanged_v4_v5_methods'],
                  'cache_contract': out['repeated_negative_review'],
                  'v5_grouped_questions': out['runs']['v5']['grouped_questions'],
                  'v5_grouped_new_needs': out['runs']['v5']['grouped_with_new_needs']}, indent=2))
