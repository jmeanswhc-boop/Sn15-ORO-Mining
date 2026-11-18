"""Record source preservation, local preflight and scripted integration comparisons."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CANDIDATE = ROOT / 'agents/Tosky_v4.py'
BASE = ROOT / 'champions/PIRATE_v1_06af6030-67a0-754e-8000-beacfd741adb/agent.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def methods(tree):
    return {node.name: node for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == 'Cairn'
            for node in cls.body if isinstance(node, ast.FunctionDef)}


def main():
    originals = json.loads((OUT / 'original-hashes.json').read_text())
    assert all(sha(ROOT / name) == digest for name, digest in originals.items())
    started = time.monotonic()
    tree = ast.parse(CANDIDATE.read_text())
    compile(tree, str(CANDIDATE), 'exec')
    parse_compile_seconds = time.monotonic() - started
    assert len([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'agent_main']) == 1
    assert not any(isinstance(n, ast.AsyncFunctionDef) and n.name == 'agent_main' for n in tree.body)
    allowed = {'__future__', 'json', 'random', 're', 'time', 'typing', 'src.agent.proxy_client'}
    denied_calls = {'eval', 'exec', '__import__', 'open', 'write_text', 'write_bytes', 'unlink', 'chmod', 'symlink_to', 'rmdir'}
    endpoints = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name in allowed for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module in allowed
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            assert name not in denied_calls
            if name == 'post_verbose':
                assert isinstance(node.args[0], ast.Constant)
                endpoints.add(node.args[0].value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not re.search(r'\b(?:t-[0-9a-f]{10}|p-[0-9a-f]{12}|w-[0-9a-f]{12})\b', node.value)
    assert endpoints == {'/inference/chat/completions', '/environment/call'}
    old, new = methods(ast.parse(BASE.read_text())), methods(tree)
    changed = sorted(n for n in old.keys() & new.keys() if ast.dump(old[n]) != ast.dump(new[n]))
    added = sorted(new.keys() - old.keys())
    graph = {n: {call.func.attr for call in ast.walk(node) if isinstance(call, ast.Call) and
                    isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and
                    call.func.value.id == 'self' and call.func.attr in new} for n, node in new.items()}
    reachable = {'run'}
    while True:
        next_set = reachable | set().union(*(graph[n] for n in reachable))
        if next_set == reachable:
            break
        reachable = next_set
    assert set(added) <= reachable
    spec = importlib.util.spec_from_file_location('tosky_checks', ROOT / 'agents/tests/test_tosky_v4.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    scenarios = {'ordinary': {}, 'explanation': {'procedure': True}, 'stockout': {'stockout': True},
                 'decline_renamed_tools': {'decline': True, 'prefix': 'renamed_'},
                 'preferences_and_explanation': {'preferences': True, 'procedure': True}}
    comparison = {}
    for label, source in [('PIRATE', BASE), ('Tosky_v4', CANDIDATE)]:
        rows = []
        for name, kwargs in scenarios.items():
            world = checks.ScriptedWorld(checks.load(source), **kwargs)
            trace = world.run()
            rows.append({'scenario': name, 'finished': world.finished,
                         'fixture_contract_satisfied': world.contract_satisfied,
                         'environment_actions': len(world.actions), 'scripted_model_requests': len(world.model_calls),
                         'diagnostic_counts': dict(Counter(r.get('kind') for r in trace))})
        comparison[label] = rows
    assert all(r['finished'] and r['fixture_contract_satisfied'] for r in comparison['Tosky_v4'])
    assert all(not before['fixture_contract_satisfied'] or after['fixture_contract_satisfied']
               for before, after in zip(comparison['PIRATE'], comparison['Tosky_v4']))
    checkpoint = OUT / 'checkpoints/06_final.py'
    checkpoint.write_bytes(CANDIDATE.read_bytes())
    checkpoint_hashes = {p.name: sha(p) for p in sorted((OUT / 'checkpoints').glob('*.py'))}
    evidence = {
        'source': str(CANDIDATE.relative_to(ROOT)), 'source_sha256': sha(CANDIDATE),
        'source_bytes': CANDIDATE.stat().st_size, 'source_lines': len(CANDIDATE.read_text().splitlines()),
        'original_sources_preserved': originals, 'checkpoints': checkpoint_hashes,
        'changed_controller_methods': changed, 'added_controller_methods': added,
        'removed_controller_methods': sorted(old.keys() - new.keys()),
        'unchanged_controller_methods': len(old.keys() & new.keys()) - len(changed),
        'new_methods_statically_reachable_from_run': sorted(set(added) & reachable),
        'local_preflight': {'passed': True, 'parse_compile_seconds': parse_compile_seconds,
                            'proxy_endpoints': sorted(endpoints),
                            'not_a_platform_acceptance_certificate': True},
        'scripted_comparison': comparison,
        'tests': {'command': 'python3 -m unittest discover -s agents/tests -p test_tosky_v4.py -v',
                  'passed': 50, 'evidence': '06-integrated-final.txt'},
        'paid_tests': 0, 'paid_inference_requests': 0,
        'observed_qualifier_accuracy_change': None, 'observed_hidden_race_accuracy_change': None,
        'budget': {'episode_envelope_usd': .050, 'thirty_invocations_allocation_usd': 1.50,
                   'ninety_invocations_allocation_usd': 4.50,
                   'caveat': 'Estimated admission, not a certified routed-price bound; scoped key remains authoritative. Extra task reruns can consume further credit.'},
        'limitations': ['Scripted fixture outcomes and request counts are not ORO scores, inference bills or latency estimates.',
                       'No live model outputs were generated; question-selection accuracy still needs controlled paid evaluation.',
                       'No private platform AST or plagiarism service was invoked.',
                       'Optional Tosky_v3 typed-review adjudication was not imported.'],
    }
    (OUT / 'build-review.json').write_text(json.dumps(evidence, indent=2) + '\n')
    provenance = {'name': 'Tosky_v4', 'source_sha256': sha(CANDIDATE),
                  'public_base': 'PIRATE v1', 'base_sha256': sha(BASE),
                  'changes': ['physical-request budget accounting', 'procedure completion and message reservation',
                              'availability and review invalidation', 'reference evidence integrity', 'focused preference questions'],
                  'validation': {'offline_checks_passed': 50, 'paid_tests': 0, 'platform_submission': False}}
    (ROOT / 'agents/Tosky_v4.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: evidence[k] for k in ('source_sha256', 'source_lines', 'source_bytes', 'unchanged_controller_methods', 'changed_controller_methods', 'local_preflight')}, indent=2))
    print(json.dumps({agent: [{k: row[k] for k in ('scenario', 'finished', 'fixture_contract_satisfied', 'environment_actions', 'scripted_model_requests')}
                             for row in rows] for agent, rows in comparison.items()}, indent=2))


if __name__ == '__main__':
    main()
