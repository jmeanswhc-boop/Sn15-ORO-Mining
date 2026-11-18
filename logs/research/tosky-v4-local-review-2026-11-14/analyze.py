"""Read-only reconciliation of supplied public practice logs; no inference calls."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CURRENT = 'local-4efab7fafd3d4cc0be7418b3ceca0e54'
FULL = 'local-c99b5b2fddef4178bd9903e1d0786a16'
OLD = ['local-cb574c03f70841abb0efec38dba1f2f0', 'local-71b9db485e8b423d986df21dacab6c2d']

def readl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(x.get('text', '') for x in value if isinstance(x, dict) and x.get('type') == 'text')
    return ''

def prompt_prefixes(path):
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if len(node.value.value) >= 150:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = node.value.value[:120]
    return out

def analyze(run, source):
    base = ROOT / 'LV_logs' / run
    summary = json.loads((base / 'summary.json').read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == summary['agent_sha256']
    latest = {}
    for row in readl(base / 'sandbox/inference_stats.jsonl'):
        latest[(row['problem_id'], row['instance_id'], row['execution_id'])] = row
    usage = {row['problem_id']: row for row in latest.values()}
    outputs = {row['problem_id']: row for row in readl(base / 'sandbox/sandbox_output.jsonl')}
    episodes = [r['episode'] for r in readl(base / 'episode_results.jsonl')]
    assert len(usage) == len(latest) == len(outputs) == len(episodes) == summary['task_count']
    prefixes = prompt_prefixes(source)
    totals = defaultdict(lambda: {'calls': 0, 'cost': 0., 'seconds': 0., 'input_tokens': 0, 'output_tokens': 0, 'cached_tokens': 0})
    rows = []
    for ep in episodes:
        u, out = usage[ep['session_id']], outputs[ep['session_id']]
        assert abs(u['inference_cost_usd'] - out['_shadow_inference_usage']['inference_cost_usd']) < 1e-9
        diag = [{k: v for k, v in d.items() if k != 'extra_info'} for d in out['dialogue']]
        calls = []
        for rec in readl(base / f"sandbox/request_log_{ep['session_id']}.jsonl"):
            if rec.get('path') != '/inference/chat/completions' or not rec.get('json_data'):
                continue
            req = rec['json_data']
            resp = rec.get('response') or {}
            if isinstance(resp, str):
                resp = json.loads(resp)
            choice = (resp.get('choices') or [{}])[0]
            msg, use = choice.get('message') or {}, resp.get('usage') or {}
            first = text_content(req['messages'][0].get('content'))
            purpose = next((name for name, prefix in prefixes.items() if first.startswith(prefix)), 'planner' if req.get('tools') else 'other')
            # Whitelist visible response fields. Exclude headers, proxy envelopes,
            # model reasoning payloads and private grading records.
            call = {'number': len(calls) + 1, 'model': req['model'], 'purpose': purpose,
                    'seconds': rec.get('duration_ms', 0) / 1000, 'usage': use,
                    'finish_reason': choice.get('finish_reason'),
                    'content': text_content(msg.get('content')), 'tool_calls': msg.get('tool_calls')}
            if purpose in ('CHECK', 'HEAR', 'GATE', 'BRIEF', 'SWAP'):
                # Prompts contain only the agent-visible request and shop records.
                call['prompt'] = '\n'.join(text_content(m.get('content')) for m in req['messages'])
            calls.append(call)
            for key in ['ALL', 'model:' + req['model'], 'purpose:' + purpose]:
                totals[key]['calls'] += 1
                totals[key]['cost'] += use.get('cost', 0)
                totals[key]['seconds'] += call['seconds']
                totals[key]['input_tokens'] += use.get('prompt_tokens', 0)
                totals[key]['output_tokens'] += use.get('completion_tokens', 0)
                totals[key]['cached_tokens'] += (use.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
        assert len(calls) == u['inference_total']
        assert abs(sum(c['usage'].get('cost', 0) for c in calls) - u['inference_cost_usd']) < 1e-8
        row = {'task': ep['task_id'], 'score': ep['verdict']['score'], 'category': ep['verdict']['category'],
               'request': ep['request_text'], 'cost': u['inference_cost_usd'], 'calls': u['inference_total'],
               'model_errors': u['inference_failed'], 'missing_costs': u['inference_cost_missing'],
               'seconds': out['execution_time'], 'steps': ep['step_count'], 'terminal': ep['terminal_reason'],
               'diagnostic_counts': dict(Counter(d.get('kind', d.get('role')) for d in diag))}
        rows.append(row)
        evidence = {**row, 'diagnostics': diag, 'calls': calls,
                    'world': [r for r in ep['ledger']['episode'] if r.get('kind') in ['action', 'observation', 'message', 'reply', 'event', 'basket', 'finish']]}
        save(HERE / 'tasks' / run / (ep['task_id'] + '.json'), evidence)
    return {'run': run, 'source_sha256': source_hash, 'world': summary['world_id'], 'pack_sha256': summary['pack_sha256'],
            'selection_mode': summary['selection_mode'], 'selection_seed': summary['selection_seed'],
            'count': len(rows), 'correct': sum(r['score'] for r in rows), 'cost': sum(r['cost'] for r in rows),
            'calls': sum(r['calls'] for r in rows), 'model_errors': sum(r['model_errors'] for r in rows),
            'missing_costs': sum(r['missing_costs'] for r in rows), 'mean_seconds': statistics.mean(r['seconds'] for r in rows),
            'max_seconds': max(r['seconds'] for r in rows), 'mean_steps': statistics.mean(r['steps'] for r in rows),
            'categories': dict(Counter(r['category'] or 'passed' for r in rows)), 'breakdown': dict(totals), 'tasks': rows}

def compare(old, new):
    a, b = [{r['task']: r for r in run['tasks']} for run in [old, new]]
    shared = sorted(a.keys() & b.keys())
    assert old['world'] == new['world'] and old['pack_sha256'] == new['pack_sha256']
    assert all(a[t]['request'] == b[t]['request'] for t in shared)
    out = {'old': old['run'], 'new': new['run'], 'shared': len(shared), 'same_world_pack_requests': True}
    for label, scores in [('both_pass', (1,1)), ('both_fail', (0,0)), ('losses', (1,0)), ('gains', (0,1))]:
        out[label] = [t for t in shared if (a[t]['score'], b[t]['score']) == scores]
    for name, side in [('old', a), ('new', b)]:
        out[name + '_matched'] = {'correct': sum(side[t]['score'] for t in shared),
                                 'cost': sum(side[t]['cost'] for t in shared),
                                 'calls': sum(side[t]['calls'] for t in shared),
                                 'mean_seconds': statistics.mean(side[t]['seconds'] for t in shared)}
    return out

if __name__ == '__main__':
    current = analyze(CURRENT, ROOT / 'agents/Tosky_v4.py')
    full = analyze(FULL, ROOT / 'agents/Tosky_v4.py')
    previous = [analyze(run, ROOT / 'agents/Tosky_v3.py') for run in OLD]
    output = {'current': current, 'full': full, 'previous': previous,
              'comparisons': [compare(old, current) for old in previous],
              'full_comparisons': [compare(old, full) for old in previous],
              'repeat_comparison': compare(current, full)}
    save(HERE / 'analysis.json', output)
    print(json.dumps({'full': {k:v for k,v in full.items() if k not in ['tasks', 'breakdown']},
                      'full_comparisons': output['full_comparisons'], 'repeat': output['repeat_comparison'],
                      'breakdown': full['breakdown']}, indent=2))
