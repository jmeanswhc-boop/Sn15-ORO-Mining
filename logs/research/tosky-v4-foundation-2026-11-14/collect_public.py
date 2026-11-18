"""Checkpointed, paced reads of released, unauthenticated public research data."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE = 'https://api.oroagents.com/v1/public/'
PACE_LOCK = threading.Lock()
NEXT_REQUEST = 0.0
VERSIONS = {
    'pirate': '06af6030-67a0-754e-8000-beacfd741adb',
    'tosky_submitted_v1': '06af5d7e-bfd4-761c-8000-575e592cb36b',
    'pooh': '06af354f-5c57-794a-8000-8b0c8413c3ae',
    'chidori': '06af2157-d650-7d40-8000-519949f07ca0',
    'giyu': '06af0c4d-6ea3-7367-8000-bf906703c8c6',
    'cairn': '06aef3d7-bf0f-779c-8000-ee9b2a46ce76',
}


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.incoming')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def get(endpoint, path, reuse=False):
    global NEXT_REQUEST
    if reuse and path.exists():
        payload = json.loads(path.read_text())
        if not payload.get('error') and not payload.get('stale'):
            return payload.get('data', payload)
    with PACE_LOCK:
        time.sleep(max(0.0, NEXT_REQUEST - time.monotonic()))
        NEXT_REQUEST = time.monotonic() + 0.4
    request = Request(BASE + endpoint, headers={'Accept': 'application/json', 'Accept-Encoding': 'gzip',
                                               'User-Agent': 'oro-public-foundation-review/1.0'})
    with urlopen(request, timeout=25) as response:
        body = response.read(16 * 1024 * 1024 + 1)
        if len(body) > 16 * 1024 * 1024:
            raise ValueError('Public response exceeded analysis limit')
        if response.headers.get('Content-Encoding') == 'gzip':
            body = gzip.decompress(body)
        payload = {'data': json.loads(body), 'fetched_at': datetime.now(timezone.utc).isoformat(),
                   'stale': False, 'error': None}
    save(path, payload)
    if payload.get('error') or payload.get('stale'):
        raise RuntimeError(str(payload.get('error') or 'stale response'))
    return payload.get('data', payload)


def metadata():
    history = get('races/history?limit=10&offset=0', ROOT/'public/history.json')
    get('top', ROOT/'public/top.json')
    get('races/current', ROOT/'public/current-race.json')
    races = [r for r in history['races'] if r['status'] == 'RACE_COMPLETE'][:6]
    for race in races:
        ident = race['race_id']
        get('races/' + ident, ROOT/f"public/races/{race['race_number']:03d}.json")
        print(json.dumps({'race': race['race_number'], 'metadata': 'saved'}), flush=True)
    for name, ident in VERSIONS.items():
        get('agent-versions/' + ident + '/status', ROOT/f'public/{name}/status.json')
        get('agent-versions/' + ident + '/runs', ROOT/f'public/{name}/runs.json')
    get('agent-versions/' + VERSIONS['pirate'] + '/problems?phase=QUALIFYING', ROOT/'public/pirate/qualifying-problems.json')
    get('agent-versions/' + VERSIONS['pirate'] + '/problems?phase=RACE', ROOT/'public/pirate/race-problems.json')
    save(ROOT/'metadata-complete.json', {'at': datetime.now(timezone.utc).isoformat(),
        'race_numbers': [r['race_number'] for r in races], 'versions': VERSIONS, 'paid_calls': 0})


def details():
    history = json.loads((ROOT/'public/history.json').read_text())['data']
    races = {r['race_id'] for r in [r for r in history['races'] if r['status'] == 'RACE_COMPLETE'][:6]}
    counts = {'saved': 0, 'gated': 0}
    for name in VERSIONS:
        runs = json.loads((ROOT/f'public/{name}/runs.json').read_text())['data']
        for run in runs:
            if not run.get('is_included'):
                continue
            if run.get('race_id') not in races and not (name == 'pirate' and run.get('phase') == 'QUALIFYING'):
                continue
            ident = run['eval_run_id']
            data = get('evaluation-runs/' + ident, ROOT/f'public/{name}/details/{ident}.json', reuse=True)
            items = data.get('items') or {}
            counts['saved'] += 1
            counts['gated'] += int(not items.get('detail_released', True))
            print(json.dumps({'agent': name, 'run': ident, 'released': items.get('detail_released'), 'items': len(items.get('items') or [])}), flush=True)
    save(ROOT/'details-complete.json', counts)


def feedback():
    project = ROOT.parents[2]
    jobs = []
    for name in ('pirate', 'tosky_submitted_v1', 'pooh'):
        for path in sorted((ROOT/f'public/{name}/details').glob('*.json')):
            detail = json.loads(path.read_text())['data']
            items = detail.get('items') or {}
            if not items.get('detail_released'):
                continue
            race_number = detail.get('race_number')
            # Some detail schemas omit the race number; the run list is authoritative.
            runs = json.loads((ROOT/f'public/{name}/runs.json').read_text())['data']
            run = next(r for r in runs if r['eval_run_id'] == detail['eval_run_id'])
            race_number = run.get('race_number')
            if name != 'pirate' and race_number not in (22, 23):
                continue
            for item in items.get('items') or []:
                if item.get('episode_result_id'):
                    jobs.append((name, run, item['episode_result_id']))
    state = {'total': len(jobs), 'available': 0, 'reused': 0, 'downloaded': 0, 'errors': [], 'paid_calls': 0}

    def one(job):
        name, run, episode = job
        destination = ROOT/f'public/{name}/feedback/{episode}.json'
        if destination.exists():
            existing = json.loads(destination.read_text())
            data = existing.get('data', existing)
            if not existing.get('error') and data.get('diagnostics_state') == 'available':
                return 'reused', True
        candidates = []
        if run.get('race_number'):
            candidates += list((project/'race-archives/suite-160').glob(
                f"race-{run['race_number']:03d}-*/agents/{VERSIONS[name]}/runs/{run['eval_run_id']}/feedback-{episode}.json"))
        old = {'tosky_submitted_v1': 'tosky', 'pooh': 'pooh'}.get(name)
        if old and run.get('race_number') == 22:
            candidates.append(ROOT.parent/f'tosky-race-022-2026-11-13/race-feedback/{old}/{episode}.json')
        for path in candidates:
            if not path.exists():
                continue
            existing = json.loads(path.read_text())
            data = existing.get('data', existing)
            if not existing.get('error') and data.get('diagnostics_state') == 'available':
                save(destination, {'data': data, 'reused_from': str(path.relative_to(project))})
                return 'reused', True
        data = get('episode-results/' + episode + '/feedback', destination)
        return 'downloaded', data.get('diagnostics_state') == 'available'

    print(json.dumps({'feedback_jobs': len(jobs), 'resumable': True}), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(one, job): job for job in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                origin, available = future.result()
                state[origin] += 1
                state['available'] += int(available)
            except Exception as error:
                state['errors'].append({'episode': futures[future][2], 'error': type(error).__name__})
            if n % 30 == 0 or n == len(jobs):
                state['checked'] = n
                save(ROOT/'feedback-progress.json', state)
                print(json.dumps({k: v for k, v in state.items() if k != 'errors'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['metadata', 'details', 'feedback'])
    args = parser.parse_args()
    {'metadata': metadata, 'details': details, 'feedback': feedback}[args.stage]()
