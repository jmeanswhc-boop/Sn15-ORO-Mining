"""Read-only analysis of user-supplied public practice traces. No service calls."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'LV_logs/local-57eb955c7dcc48dc8d62f192a5d8760e'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def episodes(run):
    return {r['episode']['task_id']: r['episode'] for r in rows(run / 'episode_results.jsonl')}


def requests(run, session):
    return [r for r in rows(run / 'sandbox' / ('request_log_' + session + '.jsonl')) if r.get('kind') == 'summary']


def text_of(messages):
    parts = []
    for message in messages:
        content = message.get('content')
        if isinstance(content, list):
            parts += [p.get('text', '') for p in content if isinstance(p, dict)]
        elif isinstance(content, str):
            parts.append(content)
    return '\n'.join(parts)


def metrics(run):
    summary = json.loads((run / 'summary.json').read_text())
    eps = episodes(run)
    sandbox = rows(run / 'sandbox/sandbox_output.jsonl')
    model = [r for e in eps.values() for r in requests(run, e['session_id']) if r['path'] == '/inference/chat/completions']
    costs = [r['response'].get('usage', {}).get('cost') for r in model]
    durations = [s['execution_time'] for s in sandbox]
    passed = sum(e['verdict']['score'] == 1 for e in eps.values())
    return {'run': run.name, 'source_sha256': summary['agent_sha256'], 'pack_sha256': summary['pack_sha256'],
            'tasks': len(eps), 'passed': passed, 'categories': dict(collections.Counter(e['verdict'].get('category') or 'passed' for e in eps.values())),
            'agent_model_requests': len(model), 'recorded_agent_usd': sum(c for c in costs if isinstance(c, (int, float))),
            'missing_agent_costs': sum(not isinstance(c, (int, float)) for c in costs),
            'agent_cost_per_correct': sum(c for c in costs if isinstance(c, (int, float))) / passed,
            'episode_seconds_median': statistics.median(durations), 'episode_seconds_p90': sorted(durations)[int((len(durations)-1)*.9)],
            'episode_seconds_max': max(durations), 'episode_seconds_sum_not_wall_time': sum(durations),
            'model_http_status': dict(collections.Counter(r['status_code'] for r in model))}


def main():
    summary = json.loads((RUN / 'summary.json').read_text())
    assert hashlib.sha256((ROOT / 'agents/ICDI_chidori_v3.py').read_bytes()).hexdigest() == summary['agent_sha256']
    eps = episodes(RUN)
    sandbox = {s['problem_id']: s for s in rows(RUN / 'sandbox/sandbox_output.jsonl')}
    problems = {p['problem_id']: p for p in rows(RUN / 'problems.jsonl')}
    model_stats = collections.defaultdict(lambda: {'calls': 0, 'cost_usd': 0.0, 'empty': 0, 'seconds': 0.0})
    task_stats = []
    all_calls = collections.Counter()
    failures_dir = OUT / 'cases'
    failures_dir.mkdir(exist_ok=True)
    for task, ep in eps.items():
        session = ep['session_id']
        sand = sandbox[session]
        trace = sand['dialogue']
        reqs = requests(RUN, session)
        inference = [r for r in reqs if r['path'] == '/inference/chat/completions']
        env = [r for r in reqs if r['path'] == '/environment/call']
        spend = sum(r['response'].get('usage', {}).get('cost', 0) for r in inference)
        kinds = collections.Counter(t.get('kind') for t in trace)
        all_calls.update(kinds)
        empty = []
        for index, r in enumerate(inference):
            body, data = r['json_data'], r['response']
            model = body['model']
            usage = data.get('usage') or {}
            ch = (data.get('choices') or [{}])[0]
            msg = ch.get('message') or {}
            stats = model_stats[model]
            stats['calls'] += 1
            stats['cost_usd'] += usage.get('cost') or 0
            stats['seconds'] += (r.get('duration_ms') or 0)/1000
            if not msg.get('content') and not msg.get('tool_calls'):
                stats['empty'] += 1
                empty.append({'inference_index': index, 'model': model, 'finish_reason': ch.get('finish_reason'), 'cap': body.get('max_tokens'),
                              'cost': usage.get('cost'), 'usage': usage, 'purpose_prefix': text_of(body.get('messages') or [])[:80]})
        record = {'task': task, 'session': session, 'score': ep['verdict']['score'], 'category': ep['verdict'].get('category'),
                  'steps': ep['steps'], 'max_steps': problems[session]['environment']['policy_view'].get('max_steps'),
                  'finished': trace[-1].get('finished') if trace else None, 'terminal_reason': ep.get('terminal_reason'),
                  'seconds': sand['execution_time'], 'agent_usd': spend, 'remaining_agent_allocation_usd': .035-spend,
                  'model_calls': len(inference), 'environment_calls': len(env), 'trace_counts': dict(kinds),
                  'empty_responses': empty, 'final_trace': trace[-1] if trace else None}
        task_stats.append(record)
        if ep['verdict']['score'] != 1:
            lines = [task + ' | ' + str(ep['verdict'].get('category')), 'REQUEST: ' + ep['request_text'],
                     'METRICS: ' + json.dumps(record, ensure_ascii=False), '\nAGENT TRACE:']
            for t in trace:
                d = {k:v for k,v in t.items() if k not in ('extra_info', 't')}
                lines.append(str(t.get('t')) + ' ' + json.dumps(d, ensure_ascii=False))
            lines.append('\nPUBLIC REQUESTS AND RESPONSES:')
            for index, r in enumerate(reqs):
                lines.append('\nREQUEST '+str(index)+' '+r['path']+' '+str(r.get('duration_ms'))+'ms')
                if r['path'] == '/environment/call':
                    lines += [json.dumps(r['json_data']['action'], ensure_ascii=False), json.dumps(r['response'], ensure_ascii=False)]
                else:
                    body = r['json_data']; data = r['response']
                    lines.append('MODEL '+body['model']+' cap='+str(body.get('max_tokens'))+' cost='+str(data.get('usage',{}).get('cost')))
                    lines.append('PROMPT PREFIX: '+text_of(body.get('messages') or [])[:120])
                    lines.append('REPLY: '+json.dumps(data.get('choices'), ensure_ascii=False))
            (failures_dir/(task+'.txt')).write_text('\n'.join(lines)+'\n')
    comparisons = []
    for previous in ('local-46df07bb11a34426b51f83c581c88575', 'local-f6809126cbb043e6a3b152a7582d982e', 'local-0e47aedee480429ba19d94c1909dd7f3'):
        run = ROOT / 'LV_logs' / previous
        old = episodes(run)
        stats = metrics(run)
        assert stats['pack_sha256'] == summary['pack_sha256']
        assert set(old) == set(eps)
        assert all(old[t]['request_text'] == eps[t]['request_text'] for t in eps)
        paired = collections.defaultdict(list)
        for t, ep in eps.items():
            a,b = old[t]['verdict']['score'], ep['verdict']['score']
            paired['gains' if b>a else 'losses' if b<a else 'both_pass' if b else 'both_fail'].append(t)
        comparisons.append({'metrics':stats, 'paired':dict(paired)})
    result = {'current': metrics(RUN), 'per_model':dict(model_stats), 'all_trace_counts':dict(all_calls),
              'tasks':task_stats, 'comparisons':comparisons, 'only_agent_billing_observed':True, 'paid_calls_in_analysis':False}
    (OUT/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('current','per_model','all_trace_counts','comparisons')},indent=2))


if __name__ == '__main__':
    main()
