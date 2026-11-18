"""Offline source checks and fixture comparisons. Does not submit or use inference."""
import ast
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from agents.tests import test_tosky_v5 as checks
SOURCE = ROOT / 'agents/Tosky_v5.py'
BASE = ROOT / 'agents/Tosky_v4.py'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def methods(tree):
    return {n.name: n for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == 'Cairn'
            for n in cls.body if isinstance(n, ast.FunctionDef)}

def main():
    originals = json.loads((OUT / 'original-hashes.json').read_text())
    assert all(sha(ROOT / name) == digest for name, digest in originals.items())
    assert SOURCE.stat().st_size < 1_000_000
    started = time.monotonic()
    tree = ast.parse(SOURCE.read_text())
    compile(tree, str(SOURCE), 'exec')
    seconds = time.monotonic() - started
    assert seconds < 30
    assert len([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'agent_main']) == 1
    assert not any(isinstance(n, ast.AsyncFunctionDef) and n.name == 'agent_main' for n in tree.body)
    allowed = {'__future__', 'json', 'random', 're', 'time', 'typing', 'src.agent.proxy_client'}
    prohibited = {'eval', 'exec', '__import__', 'open', 'write_text', 'write_bytes', 'unlink', 'chmod', 'symlink_to', 'rmdir'}
    endpoints = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import): assert all(a.name in allowed for a in n.names)
        if isinstance(n, ast.ImportFrom): assert n.module in allowed
        if isinstance(n, ast.Call):
            name = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ''
            assert name not in prohibited
            if name == 'post_verbose':
                assert isinstance(n.args[0], ast.Constant)
                endpoints.add(n.args[0].value)
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            assert not re.search(r'\b(?:t-[0-9a-f]{10}|p-[0-9a-f]{12}|w-[0-9a-f]{12})\b', n.value)
    assert endpoints == {'/inference/chat/completions', '/environment/call'}
    old, new = methods(ast.parse(BASE.read_text())), methods(tree)
    changed = sorted(n for n in old.keys() & new.keys() if ast.dump(old[n]) != ast.dump(new[n]))
    added = sorted(new.keys() - old.keys())
    graph = {n: {c.func.attr for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                 and isinstance(c.func.value, ast.Name) and c.func.value.id == 'self' and c.func.attr in new}
             for n, node in new.items()}
    reachable = {'run'}
    while True:
        bigger = reachable | set().union(*(graph[n] for n in reachable))
        if bigger == reachable: break
        reachable = bigger
    assert set(added) <= reachable
    scenarios = {'ordinary': {}, 'explanation': {'procedure': True}, 'stockout': {'stockout': True},
                 'decline_renamed_tools': {'decline': True, 'prefix': 'renamed_'},
                 'preferences_and_explanation': {'preferences': True, 'procedure': True},
                 'explanation_and_stockout': {'procedure': True, 'stockout': True}}
    comparison = {}
    for label, source in [('Tosky_v4', BASE), ('Tosky_v5', SOURCE)]:
        rows = []
        for name, args in scenarios.items():
            world = checks.ScriptedV5World(checks.load(source), **args)
            trace = world.run()
            assert world.finished and world.contract_satisfied
            assert not world.unknown_prompts
            rows.append({'scenario': name, 'finished': world.finished, 'fixture_contract_satisfied': world.contract_satisfied,
                         'environment_actions': len(world.actions), 'scripted_model_requests': len(world.model_calls),
                         'diagnostic_counts': dict(Counter(r.get('kind') for r in trace))})
        comparison[label] = rows
    for before, after in zip(comparison['Tosky_v4'], comparison['Tosky_v5']):
        assert after['environment_actions'] <= before['environment_actions']
    (OUT / 'checkpoints/07_final.py').write_bytes(SOURCE.read_bytes())
    checkpoints = sorted((OUT / 'checkpoints').glob('*.py'))
    (OUT / 'diffs').mkdir(exist_ok=True)
    for before, after in zip(checkpoints, checkpoints[1:]):
        delta = ''.join(difflib.unified_diff(before.read_text().splitlines(True), after.read_text().splitlines(True), fromfile=before.name, tofile=after.name))
        (OUT / 'diffs' / (after.stem + '.diff')).write_text(delta)
    evidence = {'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': sha(SOURCE), 'source_bytes': SOURCE.stat().st_size,
                'source_lines': len(SOURCE.read_text().splitlines()), 'original_sources_preserved': originals,
                'checkpoints': {p.name: sha(p) for p in checkpoints}, 'changed_controller_methods': changed,
                'added_controller_methods': added, 'removed_controller_methods': sorted(old.keys() - new.keys()),
                'unchanged_controller_methods': len(old.keys() & new.keys()) - len(changed),
                'local_preflight': {'passed': True, 'parse_compile_seconds': seconds, 'proxy_endpoints': sorted(endpoints),
                                    'new_methods_reachable': True, 'platform_acceptance_tested': False},
                'scripted_comparison': comparison,
                'tests': {'command': 'python3 -m unittest agents.tests.test_tosky_v5', 'top_level_methods': 38,
                          'nested_preserved_tests': 36, 'evidence': '07-final-tests.txt'},
                'paid_tests': 0, 'submissions': 0, 'measured_qualifier_score_change': None,
                'budget': {'per_invocation_usd': .050, 'thirty_allocation_usd': 1.50, 'ninety_allocation_usd': 4.50,
                           'guard_unchanged': True, 'live_price_bound_certified': False},
                'limitations': ['Scripted model replies are not measured accuracy, billing or latency.',
                                'The coat gender preference remains outside the bounded no-answer question branch.',
                                'The long-sleeved-top and first-cleanser failure predicates remain undisclosed.',
                                'The new audit adds one budget-controlled LIGHT call on parsed tasks; extraction reliability is unmeasured.',
                                'Lossless record-table packing failed the size feasibility check and was not added.']}
    (OUT / 'build-review.json').write_text(json.dumps(evidence, indent=2) + '\n')
    provenance = {'name': 'Tosky_v5', 'source_sha256': sha(SOURCE), 'parent': 'Tosky_v4', 'parent_sha256': sha(BASE),
                  'public_lineage': 'PIRATE v1', 'changes': ['quoted request coverage and product alternatives',
                  'eligible bounded preference questions', 'category-scoped property evidence',
                  'retention of proven replacement candidates and recovery action room', 'dependency-aware procedure proof reuse'],
                  'validation': {'offline_top_level_tests': 38, 'nested_preserved_tests': 36,
                                 'paid_tests': 0, 'platform_submission': False, 'measured_score_improvement': None}}
    (ROOT / 'agents/Tosky_v5.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k:evidence[k] for k in ['source_sha256','source_bytes','source_lines','changed_controller_methods','added_controller_methods','unchanged_controller_methods','local_preflight']},indent=2))
    print(json.dumps({k:[{c:r[c] for c in ['scenario','environment_actions','scripted_model_requests']} for r in v] for k,v in comparison.items()},indent=2))

if __name__ == '__main__': main()
