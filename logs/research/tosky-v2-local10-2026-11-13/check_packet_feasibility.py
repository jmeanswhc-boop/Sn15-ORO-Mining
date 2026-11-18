"""Quote a proposed format-only packet in the two recorded local denial states."""
import json
from pathlib import Path

from replay_recorded import ROOT, agent


def build_packet(decision):
    source = decision['packet'];answer = decision['answer']
    chosen = (answer.get('assignments') or [{}])[0].get('id')
    record = next(r for r in source['records'] if r['id'] == chosen)
    facts, omitted = [], []
    for scope, fields in [('listing', record['listing']), ('attribute', record['attributes'])]:
        for key, value in fields.items():
            if isinstance(value, (dict, list)) or isinstance(value, str) and len(value) > 180:
                omitted.append({'scope': scope, 'key': key})
                continue
            facts.append({'eid': 'e'+str(len(facts)+1), 'id': chosen, 'scope': scope, 'key': key, 'value': value})
    return {'operation': 'format_only', 'assignment': {'item': 1, 'id': chosen, 'quantity': 1},
            'required_slots': [{'rid': 'r'+str(i), 'item': number, 'source_requirement': need}
                               for i, (number, need) in enumerate(source['live'], 1)],
            'facts': facts, 'omitted_fields': omitted,
            'prior_assignment': (answer.get('assignments') or [{}])[0],
            'prior_requirement_results': [{k: r.get(k) for k in ('item', 'need', 'status')}
                                          for r in answer.get('requirements') or []]}


def main():
    decisions = json.loads((ROOT/'recorded-decisions.json').read_text())
    traces = json.loads((ROOT/'recorded-replay.json').read_text())
    output = []
    for trace in traces:
        denials = [r for r in trace['rejections'] if r['method'] == 'quote']
        if not denials:
            continue
        decision = next(d for d in decisions if d['session_id'] == trace['problem_id'] and d['role'] == 'cart_proof')
        payload = build_packet(decision)
        instruction = ('Repair only the representation of the prior review. Do not choose a different product or invent a new '
                       'requirement. Return JSON with the same item/id/quantity and requirement results using the supplied rid '
                       'and eid identifiers. Evidence is an array of eid strings. Preserve unknowns and mismatches; use '
                       'error when a result cannot be grounded. Do not convert indifference into a requirement. '
                       'Facts retain their actual JSON types. An omitted field is unavailable to this formatter.\n')
        denial = denials[-1]
        body = {'model': agent.LIGHT, 'messages': [{'role': 'user', 'content': instruction+json.dumps(payload, ensure_ascii=False)}],
                'max_tokens': 768, 'temperature': 0, 'reasoning': {'enabled': False}}
        budget = agent.InferenceBudget();budget.load_prices(agent.BUDGET_PRICE_CONFIG)
        budget.actual = budget.limit-denial['available_units']-denial['protected_units']
        quote, reason = budget.quote(body, denial['protected_units'])
        assert quote is not None, reason
        output.append({'session_id': trace['problem_id'], 'task_id': decision['task_id'],
                       'original_prompt_envelope': denial['prompt_envelope'],
                       'available_after_protected_reserve_usd': denial['available_units']/1e9,
                       'protected_reserve_usd': denial['protected_units']/1e9,
                       'proposed_packet_quote_usd': quote['units']/1e9,
                       'proposed_prompt_envelope': quote['prompt_estimate'], 'output_cap': quote['cap'],
                       'admitted': True, 'packet': payload,
                       'limitation': 'This verifies request admission using the existing estimator only. No new model response, terminal action or score is produced.'})
    assert len(output) == 2
    (ROOT/'packet-feasibility.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps([{k: v for k, v in row.items() if k != 'packet'} for row in output], indent=2))


if __name__ == '__main__':
    main()
