"""Offline, transport-blocked probes of recorded observations, not score tests."""
import json
from pathlib import Path
import re
import runpy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = 'local-4efab7fafd3d4cc0be7418b3ceca0e54'
helper = runpy.run_path(str(ROOT / 'agents/tests/test_tosky_v4.py'))
ns = helper['load']()

def snapshot(task, before_questions=False, run=RUN, stop_after=None):
    data = json.loads((HERE / 'tasks' / run / (task + '.json')).read_text())
    ep = helper['episode'](ns)
    brief = next(d['brief'] for d in data['diagnostics'] if d.get('kind') == 'brief')
    ep.request = data['request']
    ep.brief = brief
    ep.roles = brief['roles']
    ep.items = [ns['Item'](x['name'], x.get('shelf', ''), x['quantity'], x['musts']) for x in brief['items']]
    ep.budget = brief['budget']
    ep.msg_limit = brief['messages']
    ep.id_key, ep.stock_key, ep.price_key = 'item_pid', 'in_stock', 'price'
    tool = None
    for event in data['world']:
        if stop_after is not None and event.get('step', 0) > stop_after:
            break
        if event['kind'] == 'action':
            tool = event['tool']
            if before_questions and tool == ep.roles['ask']:
                break
        elif event['kind'] == 'observation':
            ep.absorb(ep.unspelled(tool, event['observation']))
    return ep, data

result = {}
ep, data = snapshot('t-82a9890683', True)
plan = next(x['ask'] for x in data['diagnostics'] if x.get('kind') == 'plan')
ep.ask_json = lambda *args: {'ask': plan}
lanes = ep.plan_lanes(3)
labels = [x for x in lanes[1] if x != 'color']
choice = ep.focus_question(1, labels)
assert choice == 'options', (choice, labels)
records = [ns['_fact_block'](ep.full.get(i) or {}) or {} for i in ep.item_ids(1)]
records = [r for r in records if r]
rankings = []
for label in labels:
    counts = {}
    for record in records:
        values = [str(v).strip() for k,v in record.items() if ns['_words'](k) == ns['_words'](label) and str(v).strip()]
        if len(values) == 1:
            counts[values[0]] = counts.get(values[0], 0) + 1
    total = sum(counts.values())
    coverage = total / len(records)
    discrimination = 1 - sum(n*n for n in counts.values()) / total**2 if total else 0.
    rankings.append({'label': label, 'coverage': coverage, 'discrimination': discrimination,
                     'score': coverage * discrimination, 'not_property': bool(re.search(ns['NOT_PROPERTY'], label)),
                     'values': counts})
result['watch_question'] = {'observed_records': len(records), 'remaining_lanes': labels,
                            'reproduced_choice': choice, 'rankings': rankings}

ep, data = snapshot('t-13e82f2c08')
cheapest = 'p-e90df59cbfd9'
verdict = ep.key_verdict(cheapest, 'style: wrap')
assert verdict[0] is False
result['skirt_property_conflict'] = {'record': ep.full[cheapest], 'code_verdict': verdict,
                                    'specific_property_verdict': ep.key_verdict(cheapest, 'skirt style: wrap')}

ep, data = snapshot('t-559c052a99')
chosen = 'p-904371f2b08b'
assert ep.items[0].musts == []
verdict = ep.key_verdict(chosen, 'occasion: wedding')
assert verdict[0] is False
result['mules_omission'] = {'parsed_requirements': ep.items[0].musts,
                          'chosen_record': ep.full[chosen],
                          'verdict_if_explicit_requirement_preserved': verdict}

ep, data = snapshot('t-b9e9292a88')
chosen = 'p-175e2ca9ab1e'
verdict = ep.key_verdict(chosen, 'skin benefit: acne care')
assert verdict[0] is True
result['cleanser_unresolved'] = {'chosen_record': ep.full[chosen], 'code_verdict': verdict,
                               'note': 'The observed requirement and record match; these public logs do not reveal the failed evaluator predicate.'}

ep, data = snapshot('t-70ef000b3f')
chosen = 'p-553bf50b0966'
assert ep.key_verdict(chosen, 'style: youth fashion')[0] is True
result['hoodie_incomplete_evidence'] = {'chosen_record': ep.full[chosen],
                                     'stated_style_verdict': ep.key_verdict(chosen, 'style: youth fashion'),
                                     'prior_disclosed_material_verdict': ep.key_verdict(chosen, 'material: nylon'),
                                     'note': 'Prior material preference is comparison evidence only; it was not supplied to the live v4 agent.'}

full_run = 'local-c99b5b2fddef4178bd9903e1d0786a16'
ep, data = snapshot('t-70ef000b3f', True, full_run)
plan = next(x['ask'] for x in data['diagnostics'] if x.get('kind') == 'plan')
ep.ask_json = lambda *args: {'ask': plan}
lanes = ep.plan_lanes(3)
labels = [x for x in lanes[1] if x not in ['gender', 'hoodie style']]
choice = ep.focus_question(1, labels)
assert choice == 'season'
result['full_hoodie_question'] = {'remaining_lanes': labels, 'reproduced_choice': choice,
                                  'note': 'The third-question override demotes material, which prior public replies disclosed as nylon.'}

ep, data = snapshot('t-9ab5c575f5', run=full_run)
assert ep.items[0].name == 'pants'
assert ep.owner_of('Jeans') == 0
assert ep.key_verdict('p-028af8354258', 'fit: baggy')[0] is True
result['full_category_alternative_loss'] = {'request': data['request'],
                                          'item_name': ep.items[0].name,
                                          'jeans_owner': ep.owner_of('Jeans'),
                                          'candidate': ep.full['p-028af8354258'],
                                          'fit_verdict': ep.key_verdict('p-028af8354258', 'fit: baggy')}

ep, data = snapshot('t-9fc67ee841', run=full_run)
assert ep.items[0].musts == ['collar: not crew neck']
verdict = ep.key_verdict('p-00e4e97823f5', 'pattern: logo')
assert verdict[0] is False
result['full_logo_omission'] = {'parsed_requirements': ep.items[0].musts,
                              'verdict_if_explicit_requirement_preserved': verdict}

ep, data = snapshot('t-abf1c45838', run=full_run, stop_after=33)
lost = 'p-73f0e390ac95'
ep.gone[lost] = 33
ep.unavailable.add(lost)
captured = []
def capture(prompt, *args):
    captured.append(prompt)
    return {'fits': [{'id': 'p-2464d01129d3', 'price': 67}]}
ep.ask_json = capture
ep.replacements([lost])
cheap = 'p-180b10b52cb0'
eligible = [i for i in reversed(ep.seen_order) if i != lost and not ep.sold_out(i)]
assert cheap in ep.full
assert cheap not in captured[0]
assert eligible.index(cheap) >= 25
result['full_clock_recency_loss'] = {'known_cheaper_id': cheap,
                                   'known_cheaper_price': ep.price_of(cheap),
                                   'position_in_newest_first_candidates': eligible.index(cheap) + 1,
                                   'prompt_limit': 25, 'included_in_replacement_prompt': cheap in captured[0],
                                   'record_already_opened': cheap in ep.full,
                                   'note': 'The later cheapest-row rescue only considers unopened records, so it does not restore this known candidate.'}

result['limits'] = 'No source changes, network, paid calls, simulated shopper answers, replay score or hidden evaluator access.'
(HERE / 'probes.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
