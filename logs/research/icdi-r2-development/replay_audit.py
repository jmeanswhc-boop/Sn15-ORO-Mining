"""Read-only replay of recorded public observations and model JSON; no inference."""
import collections
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from agents.tests.test_icdi_r1 import load

RUN = ROOT / 'LV_logs/local-8751a6f6bc7948028534ac90e8c7d730'
OUT = Path(__file__).resolve().parent


def main():
    analyses = json.loads((OUT / 'analysis.json').read_text())
    totals = collections.Counter()
    results = []
    for task in analyses:
        models = {}
        for version in ('r1', 'r2'):
            ns = load(ROOT / 'agents' / ('ICDI_' + version + '.py'))
            e = ns['Kyo']({'environment': {'binding': {}, 'policy_view': {'query': task['query'], 'tools': []}}})
            brief = copy.deepcopy(next(t['brief'] for t in task['selected_trace'] if t['kind'] == 'brief'))
            e.brief = brief
            e.items = [ns['Item'](i['name'], i['shelf'], i['quantity'], i['musts']) for i in brief['items']]
            e.unclear = copy.deepcopy(brief.get('unclear', []))
            e.budget = brief.get('budget')
            models[version] = ns, e
        events = []
        for line in (RUN / 'sandbox' / ('request_log_' + task['pid'] + '.jsonl')).read_text().splitlines():
            d = json.loads(line)
            if d.get('kind') != 'summary' or 'inference' not in d['path']:
                continue
            prompt = d['json_data']['messages'][0].get('content', '')
            if not isinstance(prompt, str) or not prompt.startswith('Read the new shopper reply'):
                continue
            source = prompt.split('NEW SOURCE:\n', 1)[1]
            batches = [(int(n), q, a) for n, q, a in re.findall(
                r'^\d+\. \(about item (\d+)\) Q: (.*?)\nA: (.*?)(?=\n\d+\. \(about item|\Z)', source, re.M | re.S)]
            got = models['r1'][0]['_json_obj'](d['response']['choices'][0]['message'].get('content'))
            accepted = {}
            for version, (_, e) in models.items():
                e.ask_json = lambda *args, value=got: copy.deepcopy(value)
                if batches:
                    delta = e.hear(batches)
                    accepted[version] = bool(delta)
                    e.take_heard(delta, [])
                else:
                    lines = source.splitlines()
                    e.take_note(lines)
                    accepted[version] = not e.pending_notes
                totals[version + ('_accepted' if accepted[version] else '_rejected')] += 1
            if accepted['r1'] and not accepted['r2']:
                raise AssertionError(('Previously valid response rejected', task['task']))
            events.append({'source_kind': 'reply' if batches else 'note', 'accepted': accepted})
        states = {v: {'needs': e.needs_now(), 'free': [i.free for i in e.items], 'budget': e.budget,
                      'pending_answers': len(e.pending_answers), 'pending_notes': len(e.pending_notes)}
                  for v, (_, e) in models.items()}
        preserved = states['r1'] == states['r2']
        active_preserved = all(states['r1'][k] == states['r2'][k] for k in ('needs', 'budget', 'pending_answers', 'pending_notes'))
        if task['score'] == 1 and not active_preserved:
            raise AssertionError(('Passing task state changed', task['task'], states))
        results.append({'task': task['task'], 'original_score': task['score'],
                        'same_replayed_state': preserved, 'same_active_requirements': active_preserved,
                        'events': events, 'states': states})
    result = {'limitations': 'Recorded model outputs only. Not a new benchmark score or an end-to-end world replay.',
              'totals': dict(totals),
              'passing_task_active_requirements_preserved': sum(r['original_score'] == 1 and r['same_active_requirements'] for r in results),
              'passing_task_identical_full_state': sum(r['original_score'] == 1 and r['same_replayed_state'] for r in results),
              'tasks': results}
    (OUT / 'reply-replay.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'tasks'}, indent=2))


if __name__ == '__main__':
    main()
