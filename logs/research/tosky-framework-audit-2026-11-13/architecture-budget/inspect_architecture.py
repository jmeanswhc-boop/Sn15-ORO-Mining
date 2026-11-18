"""Offline architecture, cost-envelope and client-resource characterization."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import mock_open, patch

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent
PROJECT = AUDIT.parents[2]
sys.path[:0] = [str(PROJECT), str(AUDIT)]
import probe_flows
from probe_framework import DIGEST, SOURCE, problem


def rates():
    result = {}
    for path in ROOT.glob('*-endpoints.json'):
        data = json.loads(path.read_text())['data']
        base, higher = [], []
        for endpoint in data.get('endpoints') or []:
            pricing = endpoint.get('pricing') or {}
            if pricing.get('prompt') is None or pricing.get('completion') is None:continue
            base.append({'input': Decimal(pricing['prompt']), 'output': Decimal(pricing['completion'])})
            for override in pricing.get('overrides') or []:
                higher.append({'min_prompt_tokens': override.get('min_prompt_tokens'),
                    'input': Decimal(override.get('prompt', pricing['prompt'])),
                    'output': Decimal(override.get('completion', pricing['completion']))})
        result[data['id']] = {'input_low': min(p['input'] for p in base), 'input_high': max(p['input'] for p in base),
            'output_low': min(p['output'] for p in base), 'output_high': max(p['output'] for p in base),
            'long_context_tiers': higher}
    assert len(result) == 3
    return result


def profile_flows(prices):
    results = []
    for scenario in ('healthy', 'format_failure', 'reference_dependency', 'read_procedure'):
        module = probe_flows.load();original = module.Tosky.infer;calls = []
        def recorded(self, purpose, prompt, deep=False, tokens=1800):
            before = self.model_calls
            answer = original(self, purpose, prompt, deep, tokens)
            count = self.model_calls-before
            calls.append({'purpose': purpose, 'deep': deep, 'requested_max_tokens': tokens,
                          'physical_attempts': count, 'prompt_utf8_bytes': len(prompt.encode())})
            return answer
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No live HTTP')) as blocked, \
             patch.dict('os.environ', {'SANDBOX_MODEL': ''}), patch.object(module.Tosky, 'infer', recorded):
            summary = probe_flows.run(module, scenario);blocked.assert_not_called()
        grouped = defaultdict(lambda: {'requests': 0, 'requested_output_tokens': 0})
        per_model = defaultdict(lambda: {'requests': 0, 'requested_output_tokens': 0})
        for call in calls:
            count = call['physical_attempts']
            if not count:continue
            # All four controlled flows return their interpreter result on the
            # selected model. The existing flow assertions exclude backups.
            assert count == 1
            model = module.DEEP_MODEL if call['deep'] else module.FAST_MODEL
            grouped[call['purpose']]['requests'] += count
            grouped[call['purpose']]['requested_output_tokens'] += count*call['requested_max_tokens']
            per_model[model]['requests'] += count
            per_model[model]['requested_output_tokens'] += count*call['requested_max_tokens']
        low = sum((Decimal(p['requested_output_tokens'])*prices[model]['output_low'] for model,p in per_model.items()), Decimal(0))
        high = sum((Decimal(p['requested_output_tokens'])*prices[model]['output_high'] for model,p in per_model.items()), Decimal(0))
        assert sum(x['requests'] for x in per_model.values()) == summary['model_requests']
        assert sum(x['requested_output_tokens'] for x in per_model.values()) == summary['requested_output_tokens_total']
        results.append({'scenario': scenario, 'summary': summary, 'requests_by_purpose': dict(grouped),
                       'requests_by_model': dict(per_model),
                       'full_requested_output_cap_cost_usd_before_input': [str(low),str(high)],
                       'ninety_repetitions_output_only_full_cap_usd': [str(low*90),str(high*90)],
                       'actual_billed_cost': None, 'logical_calls_without_a_transmission': sum(not c['physical_attempts'] for c in calls),
                       'limitation': 'Controlled flows. Catalog base-context rates applied to full requested output caps. Excludes input, billing adjustments and pricing changes; this is not an actual cost forecast or measured bill.'})
    return results


def client_io():
    fake_env = {'SANDBOX_MODEL': '', 'INFERENCE_ACCESS_TOKEN': 'offline-placeholder',
                'SANDBOX_PROXY_URL': 'http://offline.invalid',
                'REQUEST_LOG_FILE': '/tmp/offline-oro-request-marker.jsonl',
                'INFERENCE_STATS_FILE': '/tmp/offline-oro-stats-marker.jsonl',
                'INFERENCE_STATS_EXECUTION_ID': 'offline-resource-check',
                'PROBLEM_DATA': '{"problem_id":"offline-resource-check"}'}
    opens, bodies = [], []
    fake_file = mock_open()
    def capture_open(path, mode='r', *args, **kwargs):
        caller = inspect.currentframe().f_back
        # MagicMock dispatch adds its own frames before reaching this side effect.
        while caller is not None and Path(caller.f_code.co_filename).name == 'mock.py':
            caller = caller.f_back
        assert caller is not None
        role = 'harness_request_log' if str(path).endswith('request-marker.jsonl') else 'harness_inference_stats'
        opens.append({'role': role, 'mode': mode, 'caller_file': Path(caller.f_code.co_filename).name,
                      'caller_function': caller.f_code.co_name, 'caller_line': caller.f_lineno})
        return fake_file(path, mode, *args, **kwargs)
    def http(url, **kwargs):
        bodies.append(deepcopy(kwargs['json']))
        body = {'model': bodies[-1]['model'], 'usage': {'cost': .000001},
                'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}
        return SimpleNamespace(status_code=200, text=json.dumps(body), json=lambda: body)
    with patch.dict('os.environ', fake_env), \
         patch('requests.sessions.Session.request', side_effect=AssertionError('Live HTTP prohibited')) as blocked:
        module = probe_flows.load();first = module.Tosky(problem());second = module.Tosky(problem())
        first.questions.append({'item':1,'property':'runtime example'})
        first.unit_factors['runtime unit','other runtime unit'] = Decimal(1)
        assert not second.questions and not second.unit_factors
        with patch('src.agent.proxy_client.requests.post', side_effect=http), patch('builtins.open', side_effect=capture_open):
            assert first.infer('offline resource trace', 'Return an empty object.', tokens=64) == {}
        blocked.assert_not_called()
    assert len(bodies) == 1 and all(o['caller_file'] == 'proxy_client.py' for o in opens)
    assert set(bodies[0]) == {'model','messages','temperature','max_tokens'}
    return {'actual_http_requests':0,'actual_file_writes':0,'intercepted_http_requests':len(bodies),
            'intercepted_file_open_attempts':opens,'inference_body_keys':list(bodies[0]),
            'native_provider_tools_requested':False,'per_task_shopping_state_isolated':True,
            'interpretation':'The supplied client attempts append-only harness logging. TOSKY itself does not choose log files or persist shopping state. Paths and HTTP were intercepted by this offline test.'}


def source_map():
    raw = SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest() == DIGEST
    tree = ast.parse(raw.decode('utf-8'))
    state = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Tosky')
    infer_sites=[]
    for method in state.body:
        if not isinstance(method,ast.FunctionDef):continue
        for node in ast.walk(method):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='infer':
                keywords={kw.arg:ast.unparse(kw.value) for kw in node.keywords}
                infer_sites.append({'caller':method.name,'line':node.lineno,'purpose':ast.unparse(node.args[0]),'options':keywords})
    return {'source_sha256':DIGEST,'infer_sites':infer_sites,
            'global_transports':[n.targets[0].id for n in tree.body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('_inference','_environment')],
            'per_task_step_and_call_limits':[{'initial_world_steps':steps,'model_attempt_limit':min(90,steps*2+14)} for steps in (20,30,38,60)],
            'budget_math':{'five_usd_for_ninety_mean_usd':str(Decimal(5)/90),
                           'illustrative_episode_envelope_usd':'0.035','ninety_envelopes_usd':'3.150','headroom_usd':'1.850',
                           'illustrative_envelope_present_in_agent':False}}


def main():
    price=rates()
    output={'created_at':datetime.now(timezone.utc).isoformat(),'source':source_map(),
            'pricing':price,'controlled_profiles':profile_flows(price),'resource_trace':client_io(),
            'paid_calls':0,'agent_modified':False}
    (ROOT/'architecture-budget-resources.json').write_text(json.dumps(output,indent=2,default=str)+'\n')
    print(json.dumps({'source_sha256':DIGEST,'controlled_profiles':[{k:r[k] for k in ('scenario','full_requested_output_cap_cost_usd_before_input','ninety_repetitions_output_only_full_cap_usd')} for r in output['controlled_profiles']],
                      'resource_trace':output['resource_trace'],'paid_calls':0,'agent_modified':False},indent=2))


if __name__=='__main__':main()
