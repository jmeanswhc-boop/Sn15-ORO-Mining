"""Offline diagnostic counterfactuals using agent-visible observations, never new scores."""
import copy
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from agents.tests.test_tosky_v4 import load
V5='local-6dfac68554764b9e8f2f025d9f0a1777'
V4='local-c99b5b2fddef4178bd9903e1d0786a16'

def readl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def snapshot(task, source='Tosky_v5.py', run=V5, stop=None):
    d=json.loads((HERE/'tasks'/run/(task+'.json')).read_text())
    p=next(p for p in readl(ROOT/'LV_logs'/run/'problems.jsonl') if p['query']==d['request'])
    ns=load(ROOT/'agents'/source)
    e=ns['Cairn']({'environment':{'binding':{'offline':True},'policy_view':copy.deepcopy(p['environment']['policy_view'])}})
    e.brief=copy.deepcopy(next(x['brief'] for x in d['diagnostics'] if x.get('kind')=='brief'))
    e.roles=e.brief['roles']
    e.items=[ns['Item'](x['name'],x.get('shelf',''),x['quantity'],x['musts']) for x in e.brief['items']]
    e.budget=e.brief['budget'];e.msg_limit=e.brief['messages']
    e.id_key,e.price_key,e.stock_key='item_pid','price','in_stock'
    tool=None
    for ev in d['world']:
        if stop is not None and ev.get('step',0)>stop:break
        if ev['kind']=='action':tool=ev['tool'];e.step=ev['step']
        elif ev['kind']=='observation':e.absorb(e.unspelled(tool,ev['observation']))
    return ns,e,d,p

out={}
# Same v5 observation set and requirements, changing only the loaded source.
for task,ident,need,stop in [
    ('t-d4e2bc9e66','p-1f4aa8507466','style: sweet',13),
    ('t-d4e2bc9e66','p-338c87630a54','style: sweet',13),
    ('t-069a67a0c4','p-0ec67862ffe2','type: ready to drink',23),
    ('t-069a67a0c4','p-035f3c32e9fe','type: ready to drink',23),
]:
    results={}
    for source in ['Tosky_v4.py','Tosky_v5.py']:
        ns,e,d,p=snapshot(task,source,stop=stop)
        assert ident in e.full
        block=ns['_fact_block'](e.full[ident])
        results[source]={'verdict':e.key_verdict(ident,need),'unfit_with_recorded_requirements':e.unfit(ident),
                         'record':e.full[ident]}
    out[task+':'+ident]=results
assert out['t-d4e2bc9e66:p-1f4aa8507466']['Tosky_v4.py']['verdict'][0] is True
assert out['t-d4e2bc9e66:p-1f4aa8507466']['Tosky_v5.py']['verdict'][0] is False

ns,e,d,p=snapshot('t-069a67a0c4',stop=23)
i='p-035f3c32e9fe'
out['milk_property_naming']={need:e.key_verdict(i,need) for need in ['container: cans','packaging: cans','type: ready to drink']}

ns,e,d,p=snapshot('t-9ab5c575f5',stop=15)
i='p-028af8354258'
e.cart_lines={i:1};e.cart_touched=True;e.cart_at=13;e.cart_seen_at=15
e.cart_view=next(ev['observation'] for ev in d['world'] if ev.get('kind')=='observation' and ev.get('step')==15)
out['alternatives_handoff']={'name':e.items[0].name,'jeans_owner':e.owner_of('Jeans'),
                             'fuller_label':e.fuller_label(1,'Jeans'),'line_faults':e.line_faults(),
                             'record':e.full[i]}
assert out['alternatives_handoff']['jeans_owner']==1
assert 'Pants & Jeans' in out['alternatives_handoff']['fuller_label']
assert out['alternatives_handoff']['line_faults']

out['empty_cart_finish']={}
for source in ['Tosky_v4.py','Tosky_v5.py']:
    ns,e,d,p=snapshot('t-069a67a0c4',source,stop=34)
    e.step=34;e.max_steps=38;e.cart_lines={};e.cart_touched=True;e.cart_at=34;e.cart_seen_at=23
    e.cart_view={'lines_items':[{'item_pid':'p-0ec67862ffe2','qty':1}],'total_price_total':87}
    e.cap_seen=len(e.full)
    calls=[]
    def act(name,args):
        calls.append(name);e.step+=1
        return {'step':e.step,'finished':False,'observation':{'issue':'BARE'}}
    e.act=act
    e.finish();e.finish()
    out['empty_cart_finish'][source]={'physical_cart_empty':True,'calls':calls,'steps':e.step}
    assert calls==[e.roles['buy'],e.roles['buy']]

out['budget_recovery_feasibility']={}
for task in ['t-069a67a0c4','t-9ab5c575f5']:
    ns,e,d,p=snapshot(task)
    e.cost=d['cost']
    session=p['problem_id']
    log=readl(ROOT/'LV_logs'/V5/'sandbox'/('request_log_'+session+'.jsonl'))
    reqs=[r['json_data'] for r in log if r.get('path')=='/inference/chat/completions' and r.get('json_data',{}).get('tools')]
    body=copy.deepcopy(reqs[-1])
    full=e.quote_inference(body)
    compact=copy.deepcopy(body)
    compact['messages']=[body['messages'][0],{'role':'user','content':e.request+'\n'+e.ledger()+'\nObserved records for a bounded recovery review:\n'+e.others([],3)}]
    compact['max_tokens']=2000
    smaller=e.quote_inference(compact)
    out['budget_recovery_feasibility'][task]={'spent':e.cost,'remaining_envelope':ns['EPISODE_USD']-e.cost,
        'last_full_request_rejected_at_final_spend':full is None,'compact_request_quote':smaller,
        'full_body_bytes':len(ns['_dump'](body).encode()),'compact_body_bytes':len(ns['_dump'](compact).encode()),
        'limit':'A quotation feasibility check only; no response or episode score generated.'}
    assert full is None and smaller is not None

out['limits']='Transport blocked. Static-method and admission probes, not new evaluation scores. No model reasoning text or private grading state used.'
(HERE/'probes.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k in ['milk_property_naming','alternatives_handoff','empty_cart_finish','budget_recovery_feasibility']},indent=2))
