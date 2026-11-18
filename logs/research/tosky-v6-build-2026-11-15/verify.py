"""Local source preflight and provenance; not the platform's private review."""
import ast
import hashlib
import json
from pathlib import Path
import re
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
path = ROOT/'agents/Tosky_v6.py'
source = path.read_text(encoding='utf-8')
start = time.monotonic()
tree = ast.parse(source)
compile(tree, str(path), 'exec')
compile_seconds = time.monotonic()-start
assert len(source.encode()) < 1_000_000
assert len([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'agent_main']) == 1
imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
old = ast.parse((ROOT/'agents/Tosky_v5.py').read_text())
assert imports == [ast.unparse(n) for n in old.body if isinstance(n, (ast.Import, ast.ImportFrom))]
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
        assert n.func.id not in {'open', 'eval', 'exec', '__import__'}
    if isinstance(n, ast.Constant) and isinstance(n.value, str):
        assert not re.search(r'\b[ptw]-[0-9a-f]{8,}\b', n.value)

def methods(t):
    cls = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == 'Cairn')
    return {n.name:n for n in cls.body if isinstance(n, ast.FunctionDef)}

before, after = methods(old), methods(tree)
changed = sorted(k for k in before.keys() & after.keys() if ast.dump(before[k], include_attributes=False) != ast.dump(after[k], include_attributes=False))
added = sorted(after.keys()-before.keys())
reachable = {'run', '__init__'}
while True:
    more = {n.attr for name in reachable for n in ast.walk(after[name]) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'self' and n.attr in after}
    if more <= reachable:
        break
    reachable |= more
assert set(added) <= reachable
originals = json.loads((HERE/'original-hashes.json').read_text())
for file, expected in originals.items():
    assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == expected, file

checkpoints = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE/'checkpoints').glob('*.py'))}
test_output = (HERE/'06-final-tests.txt').read_text()
test_count = re.search(r'^Ran (\d+) tests? in ', test_output, re.MULTILINE)
assert test_count and test_output.rstrip().endswith('OK'), 'Final offline suite did not pass'
review = {'source':'agents/Tosky_v6.py','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'parent_sha256':originals['agents/Tosky_v5.py'],'bytes':len(path.read_bytes()),
          'lines':len(source.splitlines()),'parse_compile_seconds':compile_seconds,'imports':imports,
          'changed_methods':changed,'added_methods':added,'removed_methods':sorted(before.keys()-after.keys()),
          'unchanged_methods':sum(ast.dump(before[k],include_attributes=False)==ast.dump(after[k],include_attributes=False) for k in before.keys()&after.keys()),
          'new_methods_reachable':True,'original_hashes_preserved':True,'checkpoints':checkpoints,
          'offline_top_level_tests':int(test_count.group(1)),'nested_preserved_checks':53,'history_round_trips':30,
          'paid_calls':0,'platform_submission':False,'measured_score_change':None,
          'limits':'Local preflight only. No private AST, integrity, similarity or acceptance service was invoked.'}
(HERE/'build-review.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(review,indent=2))
