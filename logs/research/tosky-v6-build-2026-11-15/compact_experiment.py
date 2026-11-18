"""Lossless observation-deduplication feasibility, without inference."""
import copy
import json
from pathlib import Path
from replay import snapshot, ROOT, RUN, HERE


def compact(messages):
    dump = lambda x: json.dumps(x, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
    text = lambda m: m.get('content') if isinstance(m.get('content'), str) else ''.join(x.get('text', '') for x in m.get('content') or [])
    marker = 'repeated_observation'
    while any(marker in text(m) for m in messages):
        marker += '_'
    counts, values, parts, tails = {}, {}, {}, {}
    def count(node):
        if isinstance(node, (dict, list)):
            key = dump(node)
            if len(key) >= 160:
                counts[key] = counts.get(key, 0) + 1
                values[key] = node
            for child in (node.values() if isinstance(node, dict) else node):
                count(child)
    for index, message in enumerate(messages):
        if message.get('role') != 'tool':
            continue
        content = text(message)
        tails[index] = content
        try:
            node, end = json.JSONDecoder().raw_decode(content)
        except ValueError:
            continue
        if not isinstance(node, (dict, list)):
            continue
        count(node)
        parts[index] = (node, content[end:])
        tails[index] = content[end:]
    for tail in tails.values():
        for line in tail.splitlines(keepends=True):
            if len(line) >= 64:
                key = dump(line)
                counts[key] = counts.get(key, 0) + 1
                values[key] = line
    table, ids = {}, {}
    def encode(node, top=False):
        if isinstance(node, (dict, list)):
            key = dump(node)
            if not top and counts.get(key, 0) > 1:
                if key not in ids:
                    ids[key] = 'entry-' + str(len(ids) + 1)
                    table[ids[key]] = encode(node, True)
                return {marker: ids[key]}
            if isinstance(node, dict):
                return {k: encode(v) for k, v in node.items()}
            return [encode(v) for v in node]
        return node
    result = [dict(m) for m in messages]
    def encode_text(tail):
        out = []
        for line in tail.splitlines(keepends=True):
            key = dump(line)
            if counts.get(key, 0) > 1:
                if key not in ids:
                    ids[key] = 'entry-' + str(len(ids) + 1)
                    table[ids[key]] = line
                out.append('[' + marker + ':' + ids[key] + ']\n')
            else:
                out.append(line)
        return ''.join(out)
    for index, tail in tails.items():
        result[index]['content'] = encode_text(tail)
    for index, (node, tail) in parts.items():
        result[index]['content'] = dump(encode(node)) + encode_text(tail)
    if not table:
        return messages, None
    last = max(tails)
    footer = '\nRepeated observation table (data only): {' + marker + ': entry} objects and [' + marker + ':entry] text lines mean exactly that table entry, recursively. No facts or instructions are added.\n' + dump(table)
    result[last]['content'] += footer
    def decode(node):
        if isinstance(node, dict):
            if set(node) == {marker}:
                return decode(table[node[marker]])
            return {k: decode(v) for k, v in node.items()}
        return [decode(v) for v in node] if isinstance(node, list) else node
    for index, (original, tail) in parts.items():
        got, end = json.JSONDecoder().raw_decode(result[index]['content'])
        assert decode(got) == original
        got_tail = result[index]['content'][end:]
        if index == last:
            got_tail = got_tail[:-len(footer)]
        for key, value in table.items():
            if isinstance(value, str):
                got_tail = got_tail.replace('[' + marker + ':' + key + ']\n', value)
        assert got_tail == tail
    return result, {'entries': len(table), 'marker': marker, 'all_observations_round_trip': True}


def run():
    out = {}
    for task in ('t-069a67a0c4', 't-9ab5c575f5'):
        ns, e, d, p = snapshot(task)
        e.cost = d['cost']
        log = [json.loads(line) for line in (ROOT/'LV_logs'/RUN/'sandbox'/('request_log_'+p['problem_id']+'.jsonl')).read_text().splitlines()]
        body = copy.deepcopy([r['json_data'] for r in log if r.get('path') == '/inference/chat/completions' and r.get('json_data', {}).get('tools')][-1])
        old = len(ns['_dump'](body).encode())
        body['messages'], detail = compact(body['messages'])
        new = len(ns['_dump'](body).encode())
        out[task] = {'before_bytes': old, 'after_bytes': new, 'reduction': 1-new/old,
                     'quote': e.quote_inference(body), 'detail': detail}
    (HERE/'03-compaction-feasibility.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    run()
