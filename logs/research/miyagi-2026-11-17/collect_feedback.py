"""Resume public episode feedback collection for the explicitly selected runs."""
import gzip
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE = 'https://api.oroagents.com/v1/public/episode-results/'


def main():
    plan = json.loads((ROOT / 'feedback-plan.json').read_text())
    destination = ROOT / 'feedback'
    destination.mkdir(exist_ok=True)
    failures = []
    saved = 0
    for index, item in enumerate(plan, 1):
        path = destination / (item['episode_result_id'] + '.json')
        if path.exists() and json.loads(path.read_text()).get('diagnostics_state') == 'available':
            saved += 1
            continue
        try:
            request = Request(BASE + item['episode_result_id'] + '/feedback',
                              headers={'Accept-Encoding': 'gzip', 'Accept': 'application/json'})
            with urlopen(request, timeout=25) as response:
                raw = response.read(4 * 1024 * 1024)
                if response.headers.get('Content-Encoding') == 'gzip':
                    raw = gzip.decompress(raw)
                data = json.loads(raw)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
            saved += 1
        except Exception as error:
            failures.append({'episode_result_id': item['episode_result_id'], 'error': type(error).__name__})
        (ROOT / 'feedback-progress.json').write_text(json.dumps({'visited': index, 'saved': saved, 'planned': len(plan), 'failures': failures}, indent=2))
        if index % 15 == 0:
            print('Public feedback', index, '/', len(plan), 'saved', saved, 'errors', len(failures), flush=True)
        time.sleep(0.8)
    print('Public feedback collection finished', saved, 'errors', len(failures), flush=True)


if __name__ == '__main__':
    main()
