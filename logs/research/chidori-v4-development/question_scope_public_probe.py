"""Conditional question-scope probe using only recorded public state."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'agents/tests'))
from test_chidori_v4 import AGENTS,Client,response
spec=importlib.util.spec_from_file_location('old_replay',ROOT/'logs/research/chidori-v3-run-57eb955c/replay_states.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
a=r.load();original=a.Episode
problems={x['problem_id']:x for x in r.rows(r.RUN/'problems.jsonl')}
episodes={x['episode']['session_id']:x['episode'] for x in r.rows(r.RUN/'episode_results.jsonl')}
observed=[]
for row in r.rows(r.RUN/'sandbox/sandbox_output.jsonl'):
 episode=episodes[row['problem_id']]
 def construct(data):
  ep=original(data);base=ep.plan
  def plan(count):
   result=base(count)
   for n,item in enumerate(ep.items,1):
    lane=ep.lanes.get(n,[])
    for need in ep.live_of(item):
     prop=need.split(':',1)[0]
     if lane and a.stems(prop) and a.stems(prop)<a.stems(lane[0]):
      state={k:copy.deepcopy(v) for k,v in vars(ep).items() if not callable(v)}
      observed.append((episode['task_id'],data,n,need,lane[0],state))
   return result
  ep.plan=plan;return ep
 records=[x for x in r.rows(r.RUN/'sandbox'/('request_log_'+row['problem_id']+'.jsonl')) if x.get('kind')=='summary']
 with patch.object(a,'Episode',side_effect=construct): result=r.replay(a,problems[row['problem_id']],records,row['dialogue'])
 assert result['exact_replay']
results=[]
with patch('requests.sessions.Session.request',side_effect=AssertionError('Real network forbidden')) as blocked:
 for task,data,n,need,label,state in observed:
  for name,agent in AGENTS.items():
   ep=agent.Episode(data)
   for key,value in state.items():setattr(ep,key,copy.deepcopy(value))
   ep.clock=lambda:100;ep.room=lambda:180
   public=[]
   for ident in ep.item_listings(n):
    attrs=ep.public_attributes(ident)
    keys=[key for key in attrs if agent.plainf(key)==agent.plainf(label)]
    if keys:public.append((ident,keys[0],attrs[keys[0]]))
   assert public
   ident,key,value=public[0]
   prop=need.split(':',1)[0]
   match=re.search(re.escape(prop),ep.request,re.I);assert match
   excerpt=ep.request[match.start():].split('.',1)[0]
   answer={'same_property':True,'item':n,'label':label,'need':need,'source':excerpt,
           'evidence':[{'id':ident,'scope':'attribute','key':key,'value':value}]}
   client=Client([response(json.dumps(answer))])
   before=list(ep.remaining(n))
   with patch.object(agent.E6,'plan'),patch.object(agent,'model_client',return_value=client):ep.plan(3)
   after=ep.remaining(n)
   assert label in before and label not in after
   assert ep.items[n-1].musts==state['items'][n-1].musts
   results.append({'agent':name,'task':task,'topic':label,'request_requirement':need,'first_remaining_topic':after[0] if after else None,
                   'scripted_same_property_decision':True,'paid_or_new_accuracy_result':False,'physical_mock_attempts':len(client.posts)})
 assert not blocked.called
out=Path(__file__).parent/'question-scope-public-probe.json';out.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
