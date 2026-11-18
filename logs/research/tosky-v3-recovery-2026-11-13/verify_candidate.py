"""Unpaid source, preservation and captured-budget admission verification."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT/'agents/tests')]
from test_tosky_v3 import agent, baseline, recorded_episode, typed_answer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_check():
    path = PROJECT/'agents/Tosky_v3.py'
    before, after = [ast.parse(p.read_text()) for p in (PROJECT/'agents/ICDI_chidori_v4.py', path)]
    def methods(tree):
        return {(c.name, f.name): ast.dump(f) for c in tree.body if isinstance(c, ast.ClassDef)
                for f in c.body if isinstance(f, ast.FunctionDef)}
    original, updated = methods(before), methods(after)
    assert all(updated[key] == value for key, value in original.items())
    allowed = {'json', 'random', 're', 'time', 'typing', 'decimal', 'src.agent.proxy_client'}
    forbidden = {'eval', 'exec', '__import__', 'open', 'write_text', 'write_bytes', 'unlink', 'chmod', 'getenv'}
    for node in ast.walk(after):
        if isinstance(node, ast.Import):assert all(a.name in allowed for a in node.names)
        if isinstance(node, ast.ImportFrom):assert node.module in allowed
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            assert name not in forbidden, name
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            import re
            assert not re.search(r'\b(?:t-[0-9a-f]{10}|p-[0-9a-f]{12})\b', node.value)
    assert len([n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == 'agent_main']) == 1
    assert not any(isinstance(n, ast.AsyncFunctionDef) and n.name == 'agent_main' for n in after.body)
    assert path.stat().st_size < 1000000
    assert '/inference/models' not in path.read_text()
    assert agent.AGENT_USD_LIMIT == baseline.AGENT_USD_LIMIT == '0.035'
    assert agent.PROCEDURE_USD_RESERVE == baseline.PROCEDURE_USD_RESERVE == '0.010'
    assert agent.BUDGET_PRICE_CONFIG == baseline.BUDGET_PRICE_CONFIG
    assert agent.CALL_CAP == baseline.CALL_CAP == 60
    new_methods = [name for owner, name in updated if owner == 'ReviewBoundaryControls']
    assert 'adjudicate_review' in new_methods and 'budget_post' in new_methods
    for name in ('apply_source_audit', 'validate_cart_proof', 'property_key', 'decline_check'):
        assert not hasattr(agent.Episode, name), name
    return {'bytes': path.stat().st_size, 'sha256': sha(path), 'original_methods_preserved': len(original),
            'new_review_methods': new_methods, 'synchronous_entrypoint': True, 'local_ast_preflight': 'passed',
            'unchanged_budget_and_pricing': True, 'official_backend_admission': 'not performed',
            'limitation': 'Syntax and focused local checks are not the unpublished server AST/integrity review.'}


def preservation_check():
    sources = json.loads((ROOT/'preserved-sources.json').read_text())
    for entry in sources:assert sha(PROJECT/entry['path']) == entry['sha256']
    manifest = json.loads((ROOT.parent/'tosky-v2-local10-2026-11-13/manifest.json').read_text())
    run = PROJECT/manifest['provided_run']
    for entry in manifest['input_files']:assert sha(run/entry['path']) == entry['sha256']
    return {'agent_sources_unchanged': len(sources), 'supplied_log_files_unchanged': len(manifest['input_files'])}


def captured_budget_check():
    source = ROOT.parent/'tosky-v2-local10-2026-11-13'
    decisions = json.loads((source/'recorded-decisions.json').read_text())
    output = []
    for recorded in json.loads((source/'recorded-replay.json').read_text()):
        denials = [row for row in recorded['rejections'] if row['method'] == 'quote']
        if not denials:continue
        decision = next(d for d in decisions if d['session_id'] == recorded['problem_id'] and d['role'] == 'cart_proof')
        ep, ident = recorded_episode(decision);denial = denials[-1]
        ep.procedures_valid = denial['protected_units'] == 0;ep.budget_mode = 'procedure'
        budget = ep.inference_budget
        budget.actual = budget.limit-denial['available_units']-denial['protected_units']
        initial = budget.actual
        key = ident+'#'+str(ep.version);ep.verdicts = {key: False};ep.review_messages = {key: 'Recorded review is disputed.'}
        snapshot = ep.boundary_snapshot([ident]);answer = typed_answer(snapshot);answer['revision'] += 1
        malformed = json.loads(json.dumps(answer));malformed['candidates'][0]['needs'][0]['evidence'] = 'Prose in place of an evidence-ID array.'
        posts, quotes = [], []
        def post(path, body):
            assert path == '/inference/chat/completions'
            posts.append(body)
            reserve = budget.pending
            quotes.append(reserve)
            content = malformed if len(posts) == 1 else answer
            # Charge the entire conservative reservation, exceeding ordinary
            # measured use, to test whether the second call still fits.
            return SimpleNamespace(data={'usage': {'cost': reserve/1e9}, 'choices': [
                {'message': {'content': json.dumps(content)}, 'finish_reason': 'stop'}]}, error=None)
        with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')) as blocked, \
             patch.object(agent, 'model_client', return_value=SimpleNamespace(post_verbose=post)), \
             patch.object(ep, 'env', side_effect=AssertionError('No world action during representation correction')):
            assert ep.adjudicate_review([ident]), {'task': decision['task_id'], 'transmissions': len(posts),
                'quotes': quotes, 'trace': list(ep.trace)[-8:]}
            blocked.assert_not_called()
        assert len(posts) == budget.attempts == 2
        assert all(body['max_tokens'] == 768 for body in posts)
        assert budget.pending == 0 and budget.actual+budget.uncertain <= budget.limit-denial['protected_units']
        assert ep.budget_mode == 'procedure'
        output.append({'session_id': recorded['problem_id'], 'task_id': decision['task_id'],
                       'original_denied_prompt_estimate': denial['prompt_envelope'],
                       'available_ordinary_usd': denial['available_units']/1e9,
                       'protected_usd': denial['protected_units']/1e9,
                       'physical_transmissions': len(posts), 'output_caps': [body['max_tokens'] for body in posts],
                       'prompt_bytes': [len(body['messages'][0]['content'].encode()) for body in posts],
                       'reservation_usd': [q/1e9 for q in quotes],
                       'total_worst_reserved_usd': (budget.actual-initial)/1e9,
                       'ordinary_remaining_usd': budget.remaining(denial['protected_units'])/1e9,
                       'procedure_reserve_preserved': True,
                       'limitation': 'Captured balances and runtime records; controlled model replies and conservative local estimates. No new task score or measured inference bill.'})
    assert len(output) == 2
    return output


def main():
    result = {'source': source_check(), 'preservation': preservation_check(),
              'captured_budget_checks': captured_budget_check(),
              'ninety_task_allocation_usd': agent.money_units(agent.AGENT_USD_LIMIT)*90/1e9,
              'paid_calls': 0, 'submission_performed': False}
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
