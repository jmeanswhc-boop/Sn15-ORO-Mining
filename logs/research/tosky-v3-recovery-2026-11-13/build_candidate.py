"""Build a single-file recovery version from the immutable full v4 baseline."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
BASE = PROJECT/'agents/ICDI_chidori_v4.py'
TARGET = PROJECT/'agents/Tosky_v3.py'


def main():
    for entry in json.loads((ROOT/'preserved-sources.json').read_text()):
        assert hashlib.sha256((PROJECT/entry['path']).read_bytes()).hexdigest() == entry['sha256']
    baseline = BASE.read_text()
    marker = 'class Episode(ScopedControls, ProcedureControls,'
    assert baseline.count(marker) == 1
    output = baseline.replace(baseline.splitlines()[0], '# Tosky v3: bounded, typed correction of disputed reviews.', 1)
    output = output.replace(marker, (ROOT/'boundary.py.inc').read_text()+'\n\nclass Episode(ReviewBoundaryControls, ScopedControls, ProcedureControls,')
    ast.parse(output)
    assert len(output.encode('utf-8')) < 1000000
    TARGET.write_text(output)
    print(json.dumps({'file': str(TARGET), 'bytes': TARGET.stat().st_size,
                      'sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
