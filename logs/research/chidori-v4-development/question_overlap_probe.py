import importlib.util,sys
from pathlib import Path
from unittest.mock import patch
root=Path.cwd();sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location('r',root/'logs/research/chidori-v3-run-57eb955c/replay_states.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);a=r.load();old=a.Episode;problems={v['problem_id']:v for v in r.rows(r.RUN/'problems.jsonl')};eps={v['episode']['session_id']:v['episode'] for v in r.rows(r.RUN/'episode_results.jsonl')};found=[]
for row in r.rows(r.RUN/'sandbox/sandbox_output.jsonl'):
 e=eps[row['problem_id']]
 def make(data):
  ep=old(data);plan=ep.plan
  def wrapped(count):
   result=plan(count)
   for n,item in enumerate(ep.items,1):
    lane=ep.lanes.get(n,[])
    if not lane:continue
    for need in ep.live_of(item):
     prop,value=need.split(':',1) if ':' in need else(need,'')
     p=a.stems(prop);l=a.stems(lane[0])
     if p and p<l: found.append({'task':e['task_id'],'passed':e['verdict']['score']==1,'label':lane[0],'need':need})
   return result
  ep.plan=wrapped;return ep
 records=[v for v in r.rows(r.RUN/'sandbox'/('request_log_'+row['problem_id']+'.jsonl')) if v.get('kind')=='summary']
 with patch.object(a,'Episode',side_effect=make):res=r.replay(a,problems[row['problem_id']],records,row['dialogue'])
 assert res['exact_replay']
print(found)
