"""Resume only released race evidence through the running monitor's paced client."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
RACE = '50c9c9e5-335e-4964-bd6d-98af7178d3dd'
BASE = 'http://127.0.0.1:8765/api/'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.incoming')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def get(endpoint):
    with urlopen(BASE+endpoint, timeout=120) as response:
        payload = json.load(response)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch-seconds', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    deadline = time.monotonic()+max(0, args.watch_seconds)
    state = {'race_id': RACE, 'runs': {}, 'attempts': []}
    progress = ROOT/'race-collection-state.json'
    if progress.exists():
        state = json.loads(progress.read_text())
    runs = []
    for agent in ['tosky', 'chidori', 'pooh']:
        for run in json.loads((ROOT/(agent+'-runs.json')).read_text())['data']:
            if run.get('race_id') == RACE and run.get('is_included'):
                runs.append((agent, run['eval_run_id']))
    while True:
        jobs = []
        for agent, run in runs:
            detail_path = ROOT/'race-details'/agent/(run+'.json')
            payload = json.loads(detail_path.read_text()) if detail_path.exists() else {}
            old_items = (payload.get('data') or {}).get('items') or {}
            already_complete = (not payload.get('stale') and not payload.get('error')
                and old_items.get('detail_released') is True
                and old_items.get('expected_item_count', 0) > 0
                and old_items.get('expected_item_count') == old_items.get('persisted_item_count'))
            if not already_complete:
                payload = get('evaluation-runs/'+run)
                save(detail_path, payload)
            data = payload.get('data') or {}
            items = data.get('items') or {}
            valid = not payload.get('error') and not payload.get('stale')
            released = valid and items.get('detail_released') is True
            state['runs'][run] = {'agent': agent, 'released': released,
                'expected': items.get('expected_item_count') if released else None,
                'persisted': items.get('persisted_item_count') if released else None,
                'episodes_available': 0, 'checked_at': datetime.now(timezone.utc).isoformat()}
            if released:
                for item in items.get('items') or []:
                    ident = item.get('episode_result_id')
                    if ident:
                        jobs.append((agent, run, ident, item.get('item_id', '')))
        # Interleave matched tasks across agents; the monitor still serializes and
        # paces every upstream request through its one shared PublicAPI instance.
        jobs.sort(key=lambda job: (job[3], job[0], job[1]))
        def episode(job):
            agent, run, ident, task = job
            path = ROOT/'race-feedback'/agent/(ident+'.json')
            if path.exists():
                cached = json.loads(path.read_text())
                if not cached.get('stale') and (cached.get('data') or {}).get('diagnostics_state') == 'available':
                    return run, True
            payload = get('episode-results/'+ident+'/feedback')
            save(path, payload)
            return run, not payload.get('error') and not payload.get('stale') and (payload.get('data') or {}).get('diagnostics_state') == 'available'
        with ThreadPoolExecutor(max_workers=max(1, min(12, args.workers))) as pool:
            for n, future in enumerate(as_completed([pool.submit(episode, job) for job in jobs]), 1):
                try:
                    run, available = future.result()
                    state['runs'][run]['episodes_available'] += int(available)
                except Exception as error:
                    state.setdefault('errors', []).append(type(error).__name__)
                save(progress, state)
                if n % 18 == 0:
                    print(json.dumps({'checked': n, 'available': sum(v['episodes_available'] for v in state['runs'].values()), 'total': len(jobs)}), flush=True)
        complete = len(state['runs']) == len(runs) and all(v['released'] and isinstance(v['expected'], int) and v['expected'] > 0 and v['expected'] == v['persisted'] == v['episodes_available'] for v in state['runs'].values())
        state['complete'] = complete
        state['attempts'].append({'at': datetime.now(timezone.utc).isoformat(),
             'released_runs': sum(v['released'] for v in state['runs'].values()),
             'available_episodes': sum(v['episodes_available'] for v in state['runs'].values())})
        save(progress, state)
        print(json.dumps(state['attempts'][-1]), flush=True)
        if complete or time.monotonic() >= deadline:
            break
        time.sleep(min(60, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
