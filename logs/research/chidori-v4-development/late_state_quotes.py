import copy,importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
root=Path.cwd();sys.path.insert(0,str(root));sys.path.insert(0,str(root/'agents/tests'))
from test_chidori_v4 import AGENTS
spec=importlib.util.spec_from_file_location('reference_replay',root/'logs/research/chidori-v3-run-57eb955c/replay_states.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
a=r.load();p={v['problem_id']:v for v in r.rows(r.RUN/'problems.jsonl')};eps={v['episode']['session_id']:v['episode'] for v in r.rows(r.RUN/'episode_results.jsonl')};results=[]
for row in r.rows(r.RUN/'sandbox/sandbox_output.jsonl'):
 e=eps[row['problem_id']]
 if e['verdict'].get('category')!='did not finish':continue
 old=a.Episode;captured=[]
 def construct(data):
  ep=old(data);captured.append(ep);return ep
 records=[v for v in r.rows(r.RUN/'sandbox'/('request_log_'+row['problem_id']+'.jsonl')) if v.get('kind')=='summary']
 with patch.object(a,'Episode',side_effect=construct):
  result=r.replay(a,p[row['problem_id']],records,row['dialogue'])
 assert result['exact_replay']
 for name,newa in AGENTS.items():
  ep=newa.Episode(p[row['problem_id']])
  for key,value in vars(captured[0]).items():
   if not callable(value): setattr(ep,key,copy.deepcopy(value))
  ep.clock=lambda:100;ep.room=lambda:180
  protected=ep.inference_budget.protected if ep.duties else 0
  body={'model':newa.LIGHT,'messages':ep.repair_messages(ep.repair_reason()),'max_tokens':2048,'temperature':0,'tools':ep.offered(),'tool_choice':'auto','reasoning':{'effort':'low'}}
  quote,why=ep.inference_budget.quote(body,protected)
  assert quote, (name,e['task_id'],why)
  results.append({'agent':name,'task':e['task_id'],'reserved_usd':quote['units']/1e9,'cap':quote['cap'],'remaining_usd':ep.inference_budget.remaining(protected)/1e9})
print(json.dumps(results,indent=2));(root/'logs/research/chidori-v4-development/late-state-quotes.json').write_text(json.dumps(results,indent=2)+'\n')
