"""Build the single-file candidate from an immutable local baseline."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
BASE = PROJECT/'agents/ICDI_chidori_v4.py'
TARGET = PROJECT/'agents/Tosky_v2.py'
EXPECTED = 'bbf30300c5039956ddd18faa5c22fd27b126458339f8ae67860c87425777e898'


def main():
    baseline = BASE.read_bytes()
    assert hashlib.sha256(baseline).hexdigest() == EXPECTED
    text = baseline.decode('utf-8')
    text = text.replace(text.splitlines()[0], '# Tosky v2: source-grounded decisions and bounded evidence recovery.', 1)
    marker = 'class Episode(ScopedControls, ProcedureControls,'
    assert text.count(marker) == 1
    controls = (ROOT/'controls.py.inc').read_text()
    text = text.replace(marker, controls+'\n\nclass Episode(ToskyControls, ScopedControls, ProcedureControls,')
    ast.parse(text)
    assert len(text.encode('utf-8')) < 1000000
    TARGET.write_text(text)
    print(json.dumps({'file': str(TARGET), 'bytes': TARGET.stat().st_size,
                      'sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                      'base_unchanged': hashlib.sha256(BASE.read_bytes()).hexdigest() == EXPECTED}))


if __name__ == '__main__':
    main()
