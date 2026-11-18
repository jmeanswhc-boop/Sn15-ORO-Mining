"""Offline verification and reproducible packet-size checks; no model calls."""
import ast
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT/'agents/tests')]
from test_tosky_v2_flow import CompleteFlowTests, agent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    candidate = PROJECT/'agents/Tosky_v2.py'
    text = candidate.read_text()
    start = time.monotonic()
    tree = ast.parse(text)
    parse_seconds = time.monotonic()-start
    old_verification = json.loads((PROJECT/'logs/research/chidori-v4-development/verification.json').read_text())
    preserved = []
    for entry in old_verification['preserved_versions']+old_verification['new_versions']:
        path = Path(entry['path'])
        sha = digest(path)
        assert sha == entry['sha256'], path
        preserved.append({'path': str(path.relative_to(PROJECT)), 'sha256': sha})
    before = ast.parse((PROJECT/'agents/ICDI_chidori_v4.py').read_text())
    def methods(parsed):
        return {(c.name, f.name): ast.dump(f, include_attributes=False)
                for c in parsed.body if isinstance(c, ast.ClassDef)
                for f in c.body if isinstance(f, ast.FunctionDef)}
    original_methods, new_methods = methods(before), methods(tree)
    assert all(new_methods[k] == v for k, v in original_methods.items())
    controller = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'ToskyControls')
    entrypoints = [f for f in tree.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == 'agent_main']
    assert len(entrypoints) == 1 and isinstance(entrypoints[0], ast.FunctionDef)

    catalog = json.loads((ROOT/'current-model-catalog.json').read_text())
    upstream = json.loads((ROOT/'current-price-rows.json').read_text())
    upstream_rows = {row['id']: row for row in upstream['data']}
    prices = []
    for configured in agent.BUDGET_PRICE_CONFIG['data']:
        name = configured['id']
        assert name in catalog['models']
        current = upstream_rows[name]
        fields = ('prompt', 'completion', 'request', 'input_cache_read', 'input_cache_write')
        base = {field: current['pricing'].get(field, '0') for field in fields}
        tiers = [{'min_prompt_tokens': 0, 'pricing': base}]
        for override in current['pricing'].get('overrides', []):
            rates = dict(base)
            rates.update({field: override[field] for field in fields if field in override})
            tiers.append({'min_prompt_tokens': override['min_prompt_tokens'], 'pricing': rates})
        assert len(configured['context_tiers']) == len(tiers)
        for configured_tier, current_tier in zip(configured['context_tiers'], tiers):
            assert configured_tier['min_prompt_tokens'] == current_tier['min_prompt_tokens']
            for field in fields:
                assert Decimal(configured_tier['pricing'].get(field, '0')) == Decimal(current_tier['pricing'][field]), (name, field)
        for field in fields:
            assert Decimal(configured['pricing'].get(field, '0')) >= max(Decimal(t['pricing'][field]) for t in tiers)
        prices.append({'model': name, 'allowed': True, 'pricing_matches_saved_live_primary_row': True,
                       'context_tiers_checked': len(tiers)})

    flows = []
    for name, options in [('ordinary_order', {}), ('changed_budget_order', {'change': True}),
                          ('evidence_supported_decline', {'infeasible': True})]:
        case = CompleteFlowTests()
        case.setUp()
        try:
            state, world, packets, trace = case.simulate(**options)
        finally:
            case.doCleanups()
        quotes = []
        for index, packet in enumerate(packets):
            row = {'request': index+1, 'model': packet['model'],
                   'bytes': len(json.dumps(packet, ensure_ascii=False).encode()),
                   'requested_output_tokens': packet['max_tokens'], 'admission': {}}
            for spent in ('0', '.020', '.030'):
                budget = agent.InferenceBudget()
                budget.load_prices(agent.BUDGET_PRICE_CONFIG)
                budget.actual = agent.money_units(spent)
                quote, reason = budget.quote(packet)
                row['admission'][spent] = ({'reserved_usd': quote['units']/1e9,
                                             'output_cap': quote['cap'], 'prompt_byte_envelope': quote['prompt_estimate']}
                                            if quote else {'refused': reason})
            quotes.append(row)
        flows.append({'scenario': name, 'finished': state['finished'], 'environment_calls': len(world),
                      'inference_requests': len(packets), 'packets': quotes,
                      'limitation': 'Scripted model replies verify control flow only. Synthetic usage costs are not a spend estimate.'})
    budget = agent.InferenceBudget()
    budget.load_prices(agent.BUDGET_PRICE_CONFIG)
    budget.actual = agent.money_units('.020')
    ep_data = {'environment': {'binding': {'session_id': 'offline'},
                             'policy_view': {'query': 'Find suitable equipment.', 'tools': [], 'max_steps': 20}}}
    ep = agent.Episode(ep_data)
    compact = {'model': agent.LIGHT, 'messages': ep.compact_messages('The ordinary history no longer fits.'),
               'max_tokens': 1600, 'reasoning': {'enabled': False}}
    compact_quote, compact_reason = budget.quote(compact)
    assert compact_quote is not None, compact_reason
    financial = {'agent_limit_per_task_usd': agent.AGENT_USD_LIMIT,
                 'agent_allocation_90_tasks_usd': str(Decimal(agent.AGENT_USD_LIMIT)*90),
                 'nominal_margin_below_5_usd': str(Decimal(5)-Decimal(agent.AGENT_USD_LIMIT)*90),
                 'protected_procedure_usd_per_task': agent.PROCEDURE_USD_RESERVE,
                 'physical_request_limit_per_task': agent.CALL_CAP,
                 'current_price_rows': prices,
                 'compact_quote_after_0020_spent': compact_quote,
                 'flows': flows, 'paid_inference_calls': 0,
                 'cost_certified': False,
                 'limitation': 'Byte envelopes and saved prices are estimates, not tokenizer or routed-billing guarantees. Per-run shared inference can consume the remaining margin.'}
    (ROOT/'financial-verification.json').write_text(json.dumps(financial, indent=2)+'\n')
    stages = []
    for number in range(1, 6):
        path = ROOT/f'checkpoints/stage{number}.py'
        stages.append({'stage': number, 'sha256': digest(path),
                       'test_log': f'checkpoints/stage{number}-tests.txt'})
    result = {'verified_at': datetime.now(timezone.utc).isoformat(),
              'file': str(candidate.relative_to(PROJECT)), 'sha256': digest(candidate),
              'bytes': candidate.stat().st_size, 'ast_parse_seconds': parse_seconds,
              'synchronous_entrypoint_count': len(entrypoints),
              'new_controller_methods': sum(isinstance(f, ast.FunctionDef) for f in controller.body),
              'existing_class_methods_unchanged': len(original_methods),
              'new_controller_is_reachable': issubclass(agent.Episode, agent.ToskyControls),
              'preserved_versions': preserved, 'stages': stages,
              'no_paid_tests': True, 'no_submission': True,
              'new_accuracy_measured': False, 'official_admission_verified': False,
              'financial': 'financial-verification.json'}
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'sha256': result['sha256'], 'bytes': result['bytes'],
                      'preserved_versions': len(preserved), 'new_controller_methods': result['new_controller_methods'],
                      'flows': [{k: f[k] for k in ('scenario', 'finished', 'environment_calls', 'inference_requests')} for f in flows],
                      'ast_parse_seconds': parse_seconds}, indent=2))


if __name__ == '__main__':
    main()
