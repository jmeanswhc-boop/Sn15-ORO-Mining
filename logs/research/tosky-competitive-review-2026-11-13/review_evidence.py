"""Offline public-evidence inventory and qualifier-derived controller probes.

This research harness is not a submitted agent. It never calls a live model.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
PRIOR=ROOT.parent/'tosky-framework-audit-2026-11-13'
RACE=ROOT.parent/'tosky-race-022-2026-11-13'
CONTROLS=ROOT.parent/'tosky-v2-2026-11-13'
sys.path[:0]=[str(PROJECT),str(PRIOR)]
import probe_flows
import probe_framework as helpers

def read(path):
    data=json.loads(path.read_text())
    return data.get('data',data) if isinstance(data,dict) else data

def source_entry(path):
    raw=path.read_bytes()
    return {'path':str(path.relative_to(PROJECT)), 'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw)}

# These annotations describe public research fixtures, never production routing.
ANNOTATIONS=[
 ('5V router','Exact voltage; eligible product kind; cheapest','typed_evidence,optimality'),
 ('Halal cola','Dietary property; cheapest','typed_evidence,optimality'),
 ('Bottled multipacks','Requested category versus pack and packaging properties; cheapest','type_scope,optimality'),
 ('Bench dining set','Bench seating; discover an unstated preference','type_scope,interview'),
 ('Rimless glasses','Frame construction versus other components','type_scope,interview'),
 ('Round-toe sneakers','Toe shape versus nearby shoe properties','type_scope,interview'),
 ('Single-pack cola','Pack type versus number of units','type_scope,interview'),
 ('Pure-cotton hoodie','Exact composition versus a blend; cheapest','typed_evidence,optimality'),
 ('Formal maternity bottoms','Maternity product type and formal occasion','type_scope,interview'),
 ('Plastic dining set','Requested material and eligible set; cheapest','type_scope,optimality'),
 ('Jar hard candy','Packaging versus product subtype','type_scope,interview'),
 ('Elegant dress','Eligible item subtype; story clues for missing preferences','type_scope,interview'),
 ('Square-neck shirt','Neck property; incomplete shopper preference','type_scope,interview'),
 ('Large smart TV','Screen at least 43 inches; operating-cost story clue','typed_evidence,interview'),
 ('Walking sneakers','Walking style; exclude black; fit clue','typed_evidence,interview'),
 ('Router excluding two-pin plug','5V; exclude a plug type; cheapest','typed_evidence,optimality'),
 ('Solid-base dining set','Table-base property versus generic material/description','type_scope,interview'),
 ('Sugar-free cola','Dietary need; dynamically discover preferred form','type_scope,interview'),
 ('Polyester shirt','Material versus incidental story words','type_scope,interview'),
 ('Embroidered winter footwear','Pattern; exclude leather; intended subtype','typed_evidence,type_scope'),
 ('Regular-fit sweater/cardigan','Permitted category alternatives; fit','typed_evidence,type_scope'),
 ('Large hoodie excluding short sleeves','Size; exclusion; cheapest','typed_evidence,optimality'),
 ('Brown sweater/cardigan','Color; category alternatives; cheapest','typed_evidence,optimality'),
 ('Blue hoodie','Explicit color; avoid turning indifference into obligations','interview,type_scope'),
 ('Brown mary janes','Product type, color and excluded material','type_scope,typed_evidence'),
 ('Large-serve jam/marmalade','Pack type and allowed product alternatives','type_scope,interview'),
 ('Basic shirt','Explicit style; avoid inventing preferences from story','type_scope,interview'),
 ('Polyester sweater','Material; interpretation of subsequent shopper reply','typed_evidence,interview'),
 ('Slip-on winter footwear','Closure exclusions; eligible category; disclosed preferences','typed_evidence,interview'),
 ('Prescription lens capability','Coverage through +/-6; lens-color OR alternatives; cheapest','typed_evidence,optimality'),
]

def public_matrix():
    baseline=read(RACE/'analysis.json')
    tasks=read(RACE/'tosky-problems.json')['qualifying_tasks']
    historical={row['task']:row for row in baseline['qualifying_tasks']}
    extra=read(CONTROLS/'six-task-public-outcomes.json')
    assert len(tasks)==len(ANNOTATIONS)==30
    rows=[]
    for task,(label,obligation,risks) in zip(tasks,ANNOTATIONS):
        goal=task['goal_text']; old=historical[task['task_id']]
        assert goal==old['goal']
        rows.append({'task_id':task['task_id'],'label':label,'public_goal':goal,
                     'review_obligations':obligation,'risk_areas':risks.split(','),
                     'explicit_cheapest':'cheapest' in goal.casefold(),
                     'message_ceiling_three':'3 messages' in goal,
                     'step_ceiling_38':'38 steps' in goal,
                     'historic_passing_executions_out_of_two':{name:int(sum(old[key])) for name,key in
                        [('Tosky_v4','tosky'),('CHIDORI_v11','chidori'),('Pooh_v1','pooh')]},
                     'other_public_controls':extra.get(task['task_id']),
                     'TOSKY_new_framework_score':None})
    assert all(x['message_ceiling_three'] and x['step_ceiling_38'] for x in rows)
    return rows

def feedback_counts():
    output={}
    folders={'tosky':('qualifying-feedback','race-feedback/tosky'),
             'chidori':('chidori-qualifying-feedback','race-feedback/chidori'),
             'pooh':('pooh-qualifying-feedback','race-feedback/pooh')}
    total=0
    for label,paths in folders.items():
        output[label]={}
        for phase,folder in zip(('qualifying','race22'),paths):
            records=[read(p) for p in (RACE/folder).glob('*.json')]
            assert len(records)==(60 if phase=='qualifying' else 180)
            assert len({r['episode_result_id'] for r in records})==len(records)
            output[label][phase]={'executions':len(records),'unique_tasks':len({r['task_id'] for r in records}),
                'passed':sum(r['paid_reward']==1 for r in records),
                'outcomes':dict(Counter('passed' if r['paid_reward']==1 else r.get('failure_category') for r in records)),
                'end_reasons':dict(Counter(str(r.get('end_reason')) for r in records))}
            total+=len(records)
    assert total==720
    return output

def public_probes():
    observations=[]
    with patch('requests.sessions.Session.request',side_effect=AssertionError('No live HTTP')) as blocked:
        module=probe_flows.load()
        ep,_=helpers.make(module)
        ep.max_steps=38;ep.message_limit=3;ep.messages_used=2
        ep.questions=[{'item':1,'property':'an unsettled property','mandatory':False}]
        assert ep.question_to_ask() is None
        observations.append({'name':'unused_third_public_message','result':'blocked',
                             'messages_used':2,'message_ceiling':3,'world_steps_left':ep.left(),
                             'requires_wrong_model_judgment':False})
        cases=[
            ('pack_type_vs_count','cola','pack type: single',{'pack_size':1},'features.pack_size','1'),
            ('capability_range','prescription glasses','lens power: covers -6 through +6',
             {'power_range':'0.1 to 0.4'},'features.power_range','0.1 to 0.4'),
            ('component_color','prescription glasses','lens color: clear or multicolor',
             {'frame_color':'clear','lens_color':'brown'},'features.frame_color','clear'),
            ('negative_requirement','footwear','material: not leather',{'material':'leather'},'features.material','leather'),
        ]
        for label,role,requirement,facts,path,quote in cases:
            ep,c=helpers.make(module);ep.slots[0]=module.Slot(1,role,1,[requirement]);c.detail['features']=facts
            reply=helpers.judgment(requirement=requirement,path=path,quote=quote)
            with patch.object(ep,'infer',return_value=reply):ep.evaluate(c)
            assert c.verdict=='true' and ep.choose_basket()=={1:'A'}
            reply['matches'][0]['status']='false';c.evaluated_revision=-1
            with patch.object(ep,'infer',return_value=reply):ep.evaluate(c)
            assert c.verdict=='false' and not ep.choose_basket()
            observations.append({'name':label,'invalid_semantic_true_accepted':True,'correct_semantic_false_honored':True,
                                 'requires_wrong_model_judgment':True,'limitation':'Tests validation of a model mistake; not a prediction that a model will make it.'})
        ep,c=helpers.make(module);ep.slots[0]=module.Slot(1,'cola',1,['pack type: single'])
        c.detail['features']={'pack_type':'single'}
        with patch.object(ep,'infer',return_value=helpers.judgment(requirement='pack type: single',path='features.pack_type',quote='single')):
            ep.evaluate(c)
        assert c.verdict=='true' and ep.choose_basket()=={1:'A'}
        observations.append({'name':'correct_pack_type_positive_control','accepted':True})
        blocked.assert_not_called()
    return {'cases':observations,'assertion_scenarios':10,'actual_network_calls':0,'paid_calls':0,
            'note':'One interview counterexample, four semantic-validation counterexamples, four matching negative controls, one positive control. No benchmark score.'}

def inventory_champions():
    output=[]
    for p in sorted((PROJECT/'champions').glob('*/agent.py')):
        entry=source_entry(p);tree=ast.parse(p.read_text());entry.update(
            archive_name=p.parent.name,lines=len(p.read_text().splitlines()),
            functions_including_nested=sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in ast.walk(tree)))
        output.append(entry)
    return output

def third_message_control():
    p=CONTROLS/'public-agents/PRIMO/feedback/67dba863-89d2-40cd-90e0-4991a0565b96.json'
    record=read(p)
    calls=[{'step':event['step_index'],'args':call['args'],'observation':call['observation']}
           for event in record['interaction'] for call in event['calls'] if call['tool_name']=='contact_customer']
    assert len(calls)==3 and record['paid_reward']==1
    assert 'trestle' in str(calls[2]['observation'])
    return {'source':str(p.relative_to(PROJECT)),'task_id':record['task_id'],'passed':True,'messages':calls,
            'limitation':'Demonstrates useful third-question disclosure in one recorded passing interaction, not that the third question is always necessary.'}

def public_limit_flows():
    sys.path.insert(0,str(PRIOR/'architecture-budget'))
    import inspect_architecture
    original=probe_flows.problem
    def with_public_step_limit():
        payload=original()
        payload['environment']['policy_view']['max_steps']=38
        return payload
    with patch.object(probe_flows,'problem',with_public_step_limit):
        rows=inspect_architecture.profile_flows(inspect_architecture.rates())
    return {'world_step_limit':38,'model_attempt_limit':90,'flows':rows,
            'limitation':'Synthetic worlds with the public qualifier step allowance. Model replies are scripted. Pricing is the archived November 13 snapshot. Output-cap arithmetic is not an actual bill or score.'}

def main():
    source=source_entry(PROJECT/'agents/TOSKY.py');assert source['sha256']==helpers.DIGEST
    compile((PROJECT/'agents/TOSKY.py').read_text(),source['path'],'exec')
    matrix=public_matrix(); counts=feedback_counts()
    for row in matrix:
        for label,key in [('Tosky_v4','tosky'),('CHIDORI_v11','chidori'),('Pooh_v1','pooh')]:
            assert sum(r['historic_passing_executions_out_of_two'][label] for r in matrix)==counts[key]['qualifying']['passed']
    race_dir=PROJECT/'race-archives/suite-160/race-023-43b10a31-a0be-4d09-bf70-aeb7f61ae1f1'
    roster=read(race_dir/'roster.json');manifest=read(race_dir/'manifest.json')
    tracked=[r for r in roster['qualifiers'] if r['agent_version_id']=='06af5d7e-bfd4-761c-8000-575e592cb36b']
    assert len(tracked)==1 and tracked[0]['eliminated_at']
    prior_reports=[p for p in ROOT.parent.glob('*/*.txt') if p.name in ('report.txt','analysis.txt','update-plan.txt','champion-analysis.txt') and p.parent!=ROOT]
    output={'created_at':datetime.now(timezone.utc).isoformat(),'source':source,
            'public_qualifier_matrix':matrix,'public_qualifier_summary':{'tasks':len(matrix),
                'cheapest':sum(r['explicit_cheapest'] for r in matrix),'message_limit':3,'world_step_limit':38},
            'feedback_recount':counts,'third_message_control':third_message_control(),
            'new_offline_probes':public_probes(),'public_limit_flows':public_limit_flows(),
            'champion_source_inventory':inventory_champions(),
            'prior_analysis_inventory':[source_entry(p) for p in sorted(prior_reports)],
            'latest_cached_race':{'race':roster['race'],'tracked_entry':tracked[0],
                                'archive_updated_at':manifest['updated_at'],'race_score_published':tracked[0]['race_score'] is not None,
                                'local_filename_identity':'Prior research associates this public version with ICDI_chidori_v4; uploaded bytes are release-gated.'},
            'budgets':{'qualifying_limit_usd':2,'qualifying_tasks':30,'race_limit_usd':5,'race_tasks':90,
                       'qualifier_equal_share_usd':str(Decimal(2)/30),'race_equal_share_usd':str(Decimal(5)/90),
                       'uniform_illustrative_envelopes':[{ 'per_task_usd':str(n),'qualifier_allocation':str(n*30),'race_allocation':str(n*90),
                                                         'qualifier_headroom':str(Decimal(2)-n*30),'race_headroom':str(Decimal(5)-n*90)}
                                                       for n in (Decimal('.035'),Decimal('.040'),Decimal('.045'))],
                       'any_episode_usd_envelope_present_in_TOSKY':False},
            'live_refresh':{'attempted':True,'executed':False,'reason':'Automatic approval review rejected the broader public metadata refresh, including a hotkey query. Existing public records used instead.'},
            'paid_calls':0,'agent_modified':False}
    assert source_entry(PROJECT/'agents/TOSKY.py')==source
    (ROOT/'review-evidence.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    lines=['Public qualifier review matrix; historical successes are out of two executions. New TOSKY is unscored.','']
    for r in matrix:
        old=r['historic_passing_executions_out_of_two']
        lines.append(f"{r['task_id']} | {r['label']} | v4 {old['Tosky_v4']}/2, CHIDORI {old['CHIDORI_v11']}/2, Pooh {old['Pooh_v1']}/2 | {r['review_obligations']} | inspect: {', '.join(r['risk_areas'])}")
    (ROOT/'qualifier-matrix.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'source':source,'qualifiers':output['public_qualifier_summary'],'feedback_records':720,
                      'champion_sources':len(output['champion_source_inventory']),'offline_scenarios':10,
                      'latest_cached_entry':tracked[0],'paid_calls':0,'source_unchanged':True},indent=2))

if __name__=='__main__':main()
