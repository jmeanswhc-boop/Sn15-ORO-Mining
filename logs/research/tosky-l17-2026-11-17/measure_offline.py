"""Measure reversible packing on already-released observations, without inference."""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from agents.tests.test_tosky_l17 import load

OUT = Path(__file__).resolve().parent
ns = load()
results = []
started = time.monotonic()
for path in sorted((ROOT / 'logs/research/miyagi-2026-11-17/feedback').glob('*.json')):
    episode = json.loads(path.read_text())
    events = [{'source': call.get('tool_name'), 'step': turn.get('step_index'),
               'action': call.get('args'), 'observation': call.get('observation'),
               'done': call.get('done'), 'error_code': call.get('error_code')}
              for turn in episode.get('interaction') or [] for call in turn.get('calls') or []]
    value = {'source': 'store_observations', 'events': events}
    plain = ns['_dump'](value)
    rendered = ns['_context_text'](value)
    packed = json.loads(rendered)
    restored = ns['_unpack_context'](packed) if packed.get('format') == 'cairn-pack-v1' else packed
    assert ns['_dump'](restored) == plain
    a, b = len(plain.encode('utf-8')), len(rendered.encode('utf-8'))
    results.append({'feedback_file': path.name, 'calls': len(events), 'input_bytes': a,
                    'packed_bytes': b, 'reduction_fraction': 1 - b / max(1, a), 'round_trip': True})
summary = {'scope': 'Already-released observations only; whole-episode payloads, not complete live prompts',
           'paid_inference_calls': 0, 'episodes': len(results),
           'source_bytes': sum(x['input_bytes'] for x in results),
           'packed_bytes': sum(x['packed_bytes'] for x in results),
           'all_round_trips_exact': all(x['round_trip'] for x in results),
           'median_reduction_fraction': statistics.median(x['reduction_fraction'] for x in results),
           'min_reduction_fraction': min(x['reduction_fraction'] for x in results),
           'max_reduction_fraction': max(x['reduction_fraction'] for x in results),
           'elapsed_seconds': time.monotonic() - started}
summary['aggregate_reduction_fraction'] = 1 - summary['packed_bytes'] / summary['source_bytes']
(OUT / 'packing-measurements.json').write_text(json.dumps({'summary': summary, 'episodes': results}, indent=2) + '\n')
print(json.dumps(summary, indent=2))
