"""Reproducible, read-only calculations from saved public API responses."""
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
RACE = '50c9c9e5-335e-4964-bd6d-98af7178d3dd'


def read(name):
    return json.loads((ROOT/name).read_text())['data']


def run_metrics(runs):
    records = []
    for run in runs:
        summary = run.get('score_components_summary') or {}
        count = summary.get('task_count')
        mean = summary.get('mean_aggregate_reward')
        correct = round(count*mean) if count is not None and mean is not None else None
        metadata = run.get('sandbox_metadata') or {}
        seconds = (datetime.fromisoformat(run['completed_at'])-datetime.fromisoformat(run['claimed_at'])).total_seconds()
        records.append({'run': run['eval_run_id'], 'validator': run['validator_hotkey'],
            'reported_score': run.get('validator_score'), 'tasks': count, 'correct': correct,
            'outcomes': summary.get('outcome_counts'), 'inferences': run.get('inference_total'),
            'inference_errors': run.get('inference_failure_count'), 'usd': metadata.get('inference_cost_usd'),
            'sandbox_seconds': metadata.get('duration_seconds'), 'claim_to_completion_seconds': seconds,
            'service_versions': run.get('service_versions'), 'included': run.get('is_included')})
    return {'runs': records, 'correct': sum(r['correct'] for r in records),
        'tasks': sum(r['tasks'] for r in records),
        'score_from_reported_binary_counts': sum(r['correct'] for r in records)/sum(r['tasks'] for r in records),
        'mean_public_run_score': statistics.mean(r['reported_score'] for r in records),
        'inferences': sum(r['inferences'] for r in records),
        'inference_errors': sum(r['inference_errors'] for r in records),
        'mean_claim_to_completion_seconds': statistics.mean(r['claim_to_completion_seconds'] for r in records),
        'mean_sandbox_seconds': statistics.mean(r['sandbox_seconds'] for r in records) if all(r['sandbox_seconds'] is not None for r in records) else None,
        'mean_public_cost_usd': statistics.mean(r['usd'] for r in records) if all(r['usd'] is not None for r in records) else None}


def main():
    result = {'race_id': RACE, 'race_number': 22, 'suite_id': 160,
        'analyzed_at': datetime.now(timezone.utc).isoformat(),
        'local_time': datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'agents': {}, 'qualifying_tasks': [], 'source_audit': {},
        'public_limit': 'Race run details are usable only after detail_released is true; do not assign qualifier causes to hidden race episodes.',
        'paid_tests': False, 'agent_modified': False}
    for label in ['tosky', 'chidori', 'pooh']:
        allruns = read(label+'-runs.json')
        for run in allruns:
            fresh = ROOT/'race-details'/label/(run['eval_run_id']+'.json')
            if fresh.exists():
                detail = json.loads(fresh.read_text())['data']
                for field in ('sandbox_metadata', 'inference_failure_count', 'inference_total', 'score_components_summary'):
                    run[field] = detail.get(field)
        included = [r for r in allruns if r.get('is_included')]
        result['agents'][label] = {
            'race': run_metrics([r for r in included if r.get('race_id') == RACE]),
            'qualifying': run_metrics([r for r in included if r.get('phase') == 'QUALIFYING'])}
    a,b = (result['agents'][v] for v in ['tosky','chidori'])
    result['comparison'] = {
        'raw_advantage_pp_from_binary_counts': 100*(a['race']['score_from_reported_binary_counts']-b['race']['score_from_reported_binary_counts']),
        'race_inference_reduction_pct': 100*(1-a['race']['inferences']/b['race']['inferences']),
        'race_claim_to_completion_reduction_pct': 100*(1-a['race']['mean_claim_to_completion_seconds']/b['race']['mean_claim_to_completion_seconds']),
        'qualifying_cost_reduction_pct': 100*(1-a['qualifying']['mean_public_cost_usd']/b['qualifying']['mean_public_cost_usd'])}
    status = read('tosky-status-final.json') if (ROOT/'tosky-status-final.json').exists() else read('tosky-status.json')
    champion = read('chidori-status.json')
    window = next(w for w in status['window'] if w['race_id']==RACE)
    result['tosky_public_window'] = window
    result['tosky_overall'] = status.get('weighted_score')
    result['incumbent_public_overall'] = champion.get('weighted_score')
    result['tosky_eliminated_at'] = status.get('eliminated_at')
    result['tosky_is_active_qualifier'] = status.get('is_active_qualifier')
    result['ranking_caveat'] = 'Status-window rank ranks the raw single-race score; race_rank on the final roster ranks the Overall standings. These are different fields.'
    if (ROOT/'race-final.json').exists():
        roster = read('race-final.json')
        result['final_race_roster'] = [r for r in roster['qualifiers'] if r.get('agent_version_id') in
            {'06af5d7e-bfd4-761c-8000-575e592cb36b', '06af2157-d650-7d40-8000-519949f07ca0', '06af354f-5c57-794a-8000-8b0c8413c3ae'}]
    if (ROOT/'top-final.json').exists():
        result['confirmed_champion'] = read('top-final.json')
    result['score_precision_caveat'] = 'CHIDORI public status window rounds raw to .1945; the two verified binary run counts give 35/180=.194444... . Both representations are retained.'
    goals = {t['task_id']: t['goal_text'] for t in read('tosky-problems.json')['qualifying_tasks']}
    scores = defaultdict(lambda: defaultdict(list))
    for label in ['tosky','chidori','pooh']:
        for f in sorted(ROOT.glob(label+'-run-*.json')):
            data=json.loads(f.read_text())['data']
            if data.get('items',{}).get('detail_released'):
                for item in data['items']['items']:
                    scores[item['item_id']][label].append(item['paid_reward'])
    for task, values in sorted(scores.items()):
        result['qualifying_tasks'].append({'task':task,'goal':goals.get(task),**values})
    case_dir=ROOT/'evidence-cases';case_dir.mkdir(exist_ok=True)
    for label, folder in [('tosky','qualifying-feedback'),('chidori','chidori-qualifying-feedback'),('pooh','pooh-qualifying-feedback')]:
        records=[]
        for f in sorted((ROOT/folder).glob('*.json')):
            payload=json.loads(f.read_text());data=payload.get('data') or {}
            if data.get('diagnostics_state')!='available' or payload.get('stale'):continue
            calls=[{'step':t.get('step_index'),**c} for t in data.get('interaction',[]) for c in t.get('calls',[])]
            messages=[{'step':t.get('step_index'),'question':t.get('message'),'reply':t.get('shopper_message')}
                      for t in data.get('interaction',[]) if t.get('message')]
            record={'task':data.get('task_id'),'episode':data.get('episode_result_id'),
                    'passed':data.get('passed'),'category':data.get('failure_category'),
                    'termination':data.get('termination_code'),'end_reason':data.get('end_reason'),
                    'steps':data.get('step_count'),'messages':messages,'issues':data.get('observable_issues'),
                    'calls_by_tool':dict(Counter(c['tool_name'] for c in calls)),
                    'source':str(f.relative_to(ROOT))}
            records.append(record)
            if not record['passed'] or record['task'] in {'t-e622afe641','t-fa3dff6e42','t-6fd2a792df'}:
                stem=label+'-'+record['task']+'-'+record['episode']
                (case_dir/(stem+'.json')).write_text(json.dumps({'summary':record,'goal':goals.get(record['task']),'calls':calls},indent=2)+'\n')
                lines=[label+' '+record['task']+' '+str(record['passed'])+' '+str(record['category']),
                       'PUBLIC GOAL: '+str(goals.get(record['task']))]
                for c in calls:lines.append(str(c['step'])+' '+c['tool_name']+' '+json.dumps(c.get('args'),ensure_ascii=False)+' => '+json.dumps(c.get('observation'),ensure_ascii=False))
                (case_dir/(stem+'.txt')).write_text('\n'.join(lines)+'\n')
        result['agents'][label]['feedback']={'episodes_downloaded':len(records),
            'categories':dict(Counter(r['category'] or 'passed' for r in records)),
            'records':records}
    paths={'tosky':PROJECT/'agents/ICDI_chidori_v4.py',
           'chidori':PROJECT/'champions/CHIDORI_v11_06af2157-d650-7d40-8000-519949f07ca0/agent.py',
           'pooh':PROJECT/'challengers/Pooh_v1_06af354f-5c57-794a-8000-8b0c8413c3ae/agent.py'}
    methods={}
    for label,path in paths.items():
        entries={};tree=ast.parse(path.read_text());constants={}
        for node in tree.body:
            if isinstance(node,ast.ClassDef):
                for f in node.body:
                    if isinstance(f,ast.FunctionDef):entries[(node.name,f.name)]={'ast':ast.dump(f,include_attributes=False),'line':f.lineno,'end':f.end_lineno}
            if isinstance(node,ast.Assign):
                for name in node.targets:
                    if isinstance(name,ast.Name):
                        try:constants[name.id]=ast.literal_eval(node.value)
                        except (ValueError,TypeError):pass
        methods[label]=entries
        result['source_audit'][label]={'path':str(path.relative_to(PROJECT)),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'models':{k:constants.get(k) for k in ['LIGHT','HEAVY','SPARE']},
            'per_episode_limit_usd':constants.get('AGENT_USD_LIMIT'),
            'procedure_reserve_usd':constants.get('PROCEDURE_USD_RESERVE')}
    keys=[('E1','read_brief'),('E6','fold_reading'),('E6','read_answers'),('E5','value_words'),('E6','plan'),('G5B','lacks_values')]
    result['source_audit']['inherited_methods']=[{'class':c,'method':m,
        'unchanged_ast':methods['tosky'][(c,m)]['ast']==methods['chidori'][(c,m)]['ast'],
        'tosky_line':methods['tosky'][(c,m)]['line'],'chidori_line':methods['chidori'][(c,m)]['line']} for c,m in keys]
    result['race_evidence_collection'] = json.loads((ROOT/'race-collection-state.json').read_text()) if (ROOT/'race-collection-state.json').exists() else None
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name!='manifest.json':manifest.append({'path':str(p.relative_to(ROOT)),
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    (ROOT/'manifest.json').write_text(json.dumps({'created_at':result['analyzed_at'],
        'source_base':'https://api.oroagents.com/v1/public/',
        'uses_paced_local_api_client':True,'files':manifest},indent=2)+'\n')
    print(json.dumps({'comparison':result['comparison'],
        'race':{k:v['race'] for k,v in result['agents'].items()},
        'feedback_downloads':{k:v['feedback']['episodes_downloaded'] for k,v in result['agents'].items()},
        'files':len(manifest)},indent=2))


if __name__=='__main__':main()
