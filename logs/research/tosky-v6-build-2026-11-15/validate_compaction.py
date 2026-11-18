"""Round-trip every supplied final planner history with the actual candidate."""
import copy
import json
from replay import snapshot, ROOT, RUN, HERE, EVIDENCE

def text(message):
    value = message.get('content')
    return value if isinstance(value, str) else ''.join(x.get('text', '') for x in value or [])

out = []
tasks = json.loads((EVIDENCE/'analysis.json').read_text())['runs']['v5']['tasks']
for task in tasks:
    ns, e, d, p = snapshot(task['task'])
    log = [json.loads(line) for line in (ROOT/'LV_logs'/RUN/'sandbox'/('request_log_'+p['problem_id']+'.jsonl')).read_text().splitlines()]
    body = copy.deepcopy([r['json_data'] for r in log if r.get('path') == '/inference/chat/completions' and r.get('json_data', {}).get('tools')][-1])
    before = len(ns['_dump'](body).encode())
    original = copy.deepcopy(body['messages'])
    compact, detail = e.compact_messages(original)
    if detail:
        marker = detail['marker']
        table_message = next(i for i in range(len(compact)-1, -1, -1) if compact[i]['role'] == 'tool')
        table = json.loads(text(compact[table_message]).rsplit('\nRepeated observation table ', 1)[1].split('\n', 1)[1])
        def decode(node):
            if isinstance(node, dict):
                if set(node) == {marker}:
                    return decode(table[node[marker]])
                return {k: decode(v) for k, v in node.items()}
            return [decode(x) for x in node] if isinstance(node, list) else node
        def expand(value):
            for key, part in table.items():
                if isinstance(part, str):
                    value = value.replace('['+marker+':'+key+']\n', part)
            return value
        for index, (old, new) in enumerate(zip(original, compact)):
            assert {k:v for k,v in old.items() if k != 'content'} == {k:v for k,v in new.items() if k != 'content'}
            if old['role'] != 'tool':
                assert old == new
                continue
            value = text(new)
            if index == table_message:
                value = value.rsplit('\nRepeated observation table ', 1)[0]
            try:
                old_node, old_end = json.JSONDecoder().raw_decode(text(old))
            except ValueError:
                assert expand(value) == text(old)
                continue
            new_node, new_end = json.JSONDecoder().raw_decode(value)
            assert decode(new_node) == old_node
            assert expand(value[new_end:]) == text(old)[old_end:]
    else:
        assert compact == original
    e.cost, e.in_recovery = d['cost'], True
    full_quote = e.quote_inference(body)
    body['messages'] = compact
    compact_quote = e.quote_inference(body)
    if task['task'] in ('t-069a67a0c4', 't-9ab5c575f5'):
        assert full_quote is None and compact_quote is not None
    out.append({'task': task['task'], 'before_bytes': before, 'after_bytes': len(ns['_dump'](body).encode()),
                'round_trip': True, 'full_quote': full_quote, 'compact_quote': compact_quote})
(HERE/'03-public-admission.json').write_text(json.dumps({'histories':out,'limit':'Lossless data and affordability checks; no live model or task score.'},indent=2)+'\n')
print(json.dumps({'histories_verified':len(out),'unfinished':[r for r in out if r['task'] in ('t-069a67a0c4','t-9ab5c575f5')]},indent=2))
