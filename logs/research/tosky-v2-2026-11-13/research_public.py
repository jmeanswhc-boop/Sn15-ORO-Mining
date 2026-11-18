"""Bounded, resumable public qualifying-control collection; never executes agents."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
TASKS = {'t-20fcb00ff0', 't-3254025544', 't-6fd2a792df', 't-e622afe641', 't-fa3dff6e42', 't-fc0b16b797'}
VERSIONS = {
    'PRIMO': '06adca83-c5c3-7107-8000-74a56027bedc',
    'Four_Leaf': '06ae3359-6ceb-721f-8000-8ec2600c3b53',
    'Hunter': '06ae7277-714a-7ff2-8000-98ecf07e63fc',
    'Nagini': '06adf7ad-feb5-7d4e-8000-759b13868223',
    'KING': '06adb8ab-2e51-7fbd-8000-df0d87e3d5bd',
    'Abacus_r12': '06ae3256-d258-7ed0-8000-248920e28891',
}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.incoming')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def get(path, endpoint):
    if path.exists():
        cached = json.loads(path.read_text())
        if not cached.get('stale') and not cached.get('error') and cached.get('data') is not None:
            return cached
    with urlopen('http://127.0.0.1:8765/api/'+endpoint, timeout=180) as response:
        data = json.load(response)
    save(path, data)
    return data


def main():
    root = ROOT/'public-agents'
    def runs(job):
        name, version = job
        return name, version, get(root/name/'runs.json', 'agent-versions/'+version+'/runs')
    run_jobs = []
    summary = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(runs, job) for job in VERSIONS.items()]):
            name, version, payload = future.result()
            selected = [r for r in payload.get('data') or [] if r.get('phase') == 'QUALIFYING' and r.get('is_included')]
            summary[name] = {'version': version, 'included_qualifying_runs': len(selected)}
            run_jobs += [(name, r['eval_run_id']) for r in selected]
            print(name, 'qualifying runs', len(selected), flush=True)
    def detail(job):
        name, run = job
        return name, run, get(root/name/'run-details'/(run+'.json'), 'evaluation-runs/'+run)
    feedback_jobs = []
    controls = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(detail, job) for job in run_jobs]):
            name, run, payload = future.result()
            data = payload.get('data') or {}
            index = data.get('items') or {}
            items = index.get('items') or []
            matched = [r for r in items if r.get('item_id') in TASKS]
            summary[name].setdefault('runs', []).append({'run': run, 'released': index.get('detail_released'),
                'environment': data.get('environment'), 'matched_tasks': len(matched), 'count': len(items),
                'sandbox_metadata': data.get('sandbox_metadata')})
            for item in matched:
                controls.append({'agent': name, 'version': VERSIONS[name], 'run': run, **item})
                feedback_jobs.append((name, item['episode_result_id']))
            print(name, run, 'matched target tasks', len(matched), flush=True)
    save(ROOT/'public-controls-index.json', {'collected_at': datetime.now(timezone.utc).isoformat(), 'agents': summary, 'controls': controls})
    def feedback(job):
        name, episode = job
        return name, get(root/name/'feedback'/(episode+'.json'), 'episode-results/'+episode+'/feedback')
    with ThreadPoolExecutor(max_workers=6) as pool:
        for n, future in enumerate(as_completed([pool.submit(feedback, job) for job in feedback_jobs]), 1):
            name, data = future.result()
            d = data.get('data') or {}
            if d.get('diagnostics_state') != 'available':
                print(name, 'unavailable feedback', flush=True)
            if n % 12 == 0:
                print('feedback downloaded', n, '/', len(feedback_jobs), flush=True)
    sys.path.insert(0, str(PROJECT/'mining-monitor'))
    from server import PublicAPI
    from race_archive import RaceArchive
    from registry import unpack_source
    api = PublicAPI()
    source_states = {}
    for name, version in VERSIONS.items():
        destination = root/name/'source'
        provenance = destination/'provenance.json'
        if provenance.exists():
            source_states[name] = json.loads(provenance.read_text())
            continue
        try:
            metadata = api.artifact_url(version)
            destination.mkdir(parents=True, exist_ok=True)
            bundle = destination/'released-source.bin'
            RaceArchive.fetch_artifact(metadata['download_url'], bundle)
            entry, members = unpack_source(bundle.read_bytes(), destination)
            source_states[name] = {'version': version, 'state': 'downloaded', 'entry': entry,
                'members': members, 'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
                'collected_at': datetime.now(timezone.utc).isoformat()}
            save(provenance, source_states[name])
        except Exception as error:
            source_states[name] = {'version': version, 'state': 'release_gated' if getattr(error, 'release_gated', False) else 'unpublished' if getattr(error, 'not_published', False) else 'error', 'error': str(error)[:160]}
        save(ROOT/'public-source-state.json', source_states)
        print('source', name, source_states[name]['state'], flush=True)
        time.sleep(4)
    save(ROOT/'public-source-state.json', source_states)


if __name__ == '__main__':
    main()
