"""Inventory released race-18 sources without executing downloaded code."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
RACE = PROJECT/'race-archives/suite-160/race-018-f3625ba0-57a3-44d8-b318-63b9ec8e19f5'


def functions(tree):
    out = {}
    def walk(nodes, scope=''):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                walk(node.body, scope + node.name + '.')
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[scope + node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    walk(tree.body)
    return out


def main():
    manifest = json.loads((RACE/'manifest.json').read_text())
    (ROOT/'race18-release-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    pirate = json.loads((ROOT/'ast-comparison.json').read_text())['pirate_functions']
    target = {k: v['hash'] for k, v in pirate.items()}
    rows, memo = [], {}
    for n, participant in enumerate(manifest['participants'], 1):
        row = {k: participant.get(k) for k in ['agent_version_id', 'agent_name', 'version_number', 'code_state', 'code_available_at', 'code_file', 'code_sha256']}
        result = participant.get('result') or {}
        window = next((w for w in result.get('window') or [] if w['race_id'] == manifest['race']['race_id']), {})
        row.update(raw_score=result.get('race_score'), raw_rank=window.get('rank'), overall=result.get('weighted_score'))
        if participant.get('code_file'):
            path = RACE/participant['code_file']
            if path.exists():
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                assert digest == participant['code_sha256'], str(path)
                if digest not in memo:
                    tree = ast.parse(raw.decode('utf-8'))
                    index = functions(tree)
                    memo[digest] = {'bytes': len(raw), 'lines': len(raw.splitlines()), 'functions': len(index),
                        'pirate_exact_method_bodies': sum(index.get(k) == v for k, v in target.items()),
                        'pirate_common_names': len(set(index)&set(target)),
                        'branch_syntax_nodes': sum(isinstance(x, (ast.If, ast.For, ast.While, ast.Try, ast.IfExp)) for x in ast.walk(tree))}
                row.update(memo[digest])
                row['verified'] = True
        rows.append(row)
        if n % 25 == 0 or n == len(manifest['participants']):
            (ROOT/'released-source-scan.json').write_text(json.dumps({'at': datetime.now(timezone.utc).isoformat(),
                'checked': n, 'expected': len(manifest['participants']), 'unique_source_hashes': len(memo),
                'rows': rows, 'note': 'AST identity establishes shared code, not authorship or direction of derivation.'}, indent=2) + '\n')
            print(json.dumps({'sources_checked': n, 'total': len(manifest['participants']), 'unique_hashes': len(memo)}), flush=True)
    print(json.dumps({'closest': sorted(rows, key=lambda r: r.get('pirate_exact_method_bodies', 0), reverse=True)[:8]}, indent=2))


if __name__ == '__main__':
    main()
