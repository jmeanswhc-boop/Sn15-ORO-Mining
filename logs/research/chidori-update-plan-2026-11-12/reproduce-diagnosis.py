import ast,pathlib,json,collections,types,re,statistics,hashlib
R=pathlib.Path('/home/vboxuser/Documents/oro/logs/research/chidori-update-plan-2026-11-12');S=pathlib.Path('/home/vboxuser/Documents/oro/logs/research/champion-lineage-2026-11-12/source-06af2157-d650-7d40-8000-519949f07ca0.py');t=ast.parse(S.read_text());cases=[]
def method(cls,name):
 c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls);return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
def compile_method(cls,name,namespace):
 n=method(cls,name);n.decorator_list=[];mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(S),'exec'),namespace);return namespace[name]
ns={'GATE_ROUNDS':3,'GATE_PROMPT':'Perform each required procedure before STAGE.','LIGHT':'mock-light','HEAVY':'mock-heavy','dump':lambda x:json.dumps(x,sort_keys=True),'listify':lambda x:x if isinstance(x,list) else [],'count_of':lambda x:x if isinstance(x,int) else None}
gate=compile_method('E7','gate',ns)
def state(reply):
 x=types.SimpleNamespace(duties=[{'stage':'order','text':'Obtain approval for the final basket','messages':1}],finished=False,request='Get approval before ordering.',roles={'buy':'buy','refuse':'decline'},done_log=[],done_steps=[],changes=[],msg_limit=3,msg_used=0,cart_changed_at=0,gate_passes=set(),version=0,schemas={})
 x.gate_due=lambda stage:True;x.tool_lines=lambda exclusions:[];x.ledger=lambda:'One item';x.cart_text=lambda:'current basket';x.cart_mark=lambda:'basket-A';x.left=lambda:5;x.json_ask=lambda *args:reply;return x
for name,reply in [('missing model response',None),('empty model object',{}),('explicitly unproven duty without message tool',{'steps':[{'step':'Obtain approval for the final basket','done':False,'lines':[]}],'call':None})]:
 x=state(reply);allowed,_=gate(x,'order');cases.append({'scenario':name,'returned_allowed':allowed,'cached_pass':bool(x.gate_passes),'line':4689});assert allowed
# Verify forced finish ignores a negative gate result.
ns2={'CLOSEST_CLOCK':215,'shape_args':lambda args,schema:(args,[])};finish=compile_method('G6B','finish',ns2)
x=types.SimpleNamespace(finished=False,store_broke=False,items=[object()],roles={'buy':'buy'},fit_pairs=[],cart_touched=True,dropped_ids=set(),checkouts=0,refused_at=-1,cart_changed_at=0,brief={'buy_args':{}},schemas={'buy':{}},trace=[])
x.left=lambda:5;x.fresh=lambda:True;x.cart_ids=lambda:['synthetic-listing'];x.reread=lambda:None;x.gate=lambda stage:(False,'Required approval is still missing');calls=[]
def env(tool,args):calls.append(tool);x.finished=True;return {'finished':True}
x.env=env;finish(x);cases.append({'scenario':'forced finish with negative procedure gate','environment_calls':calls,'line':10094});assert calls==['buy']
# Confirm the source has no choose-stage gate invocation.
gate_calls=[n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='gate'];choose=[n.lineno for n in gate_calls if n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='choose'];cases.append({'scenario':'choose-stage gate references','lines':choose});assert not choose
# Evaluate the actual final reserve assignment with two independent order messages.
read=method('E1','read_brief');reserve=next(n for n in ast.walk(read) if isinstance(n,ast.Assign) and any(isinstance(a,ast.Attribute) and a.attr=='msg_reserve' for a in n.targets));x=types.SimpleNamespace(msg_limit=3);exec(compile(ast.fix_missing_locations(ast.Module(body=[reserve],type_ignores=[])),str(S),'exec'),{'self':x,'c':[1,1]});cases.append({'scenario':'two independent one-message obligations','reserved_messages':x.msg_reserve,'required_if_separate':2,'line':1952});assert x.msg_reserve==1
manifest=json.load(open(R/'feedback-manifest.json'));counts=collections.Counter();byrun=collections.defaultdict(collections.Counter);bytask=collections.defaultdict(list);messages=collections.defaultdict(list)
for item in manifest:
 p=R/('feedback-'+item['episode']+'.json');f=json.load(open(p));assert f['diagnostics_state']=='available';category=f.get('failure_category') or 'passed';assert f['passed'] is (category=='passed');assert f['paid_reward']==item['reward'];assert f['task_id']==item['task'];assert f['query'] is None;counts[category]+=1;byrun[item['run']][category]+=1;bytask[item['task']].append(category)
 sent=[c for st in f.get('interaction',[]) for c in st.get('calls',[]) if any(w in c.get('tool_name','').lower() for w in ['message','shopper','customer','write'])];messages[category].append(len(sent))
assert len(manifest)==180 and len({r['episode'] for r in manifest})==180
assert len(byrun)==2 and all(sum(c.values())==90 for c in byrun.values())
assert len(bytask)==90 and all(len(c)==2 for c in bytask.values())
assert hashlib.sha256(S.read_bytes()).hexdigest()=='b32f88b0ef42100f10d9ad29e3019a54dd79dfaefe8ded2b72a88772f9247bf2'
runmeta=[r for r in json.load(open(R/'runs.json')) if r.get('race_number')==21 and r.get('is_included')]
summary={'source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'episodes':len(manifest),'distinct_tasks':len(bytask),'categories':dict(counts),'by_run':{k:dict(v) for k,v in byrun.items()},'process_failure_fraction_of_failed_episodes':counts['process_requirement_missed']/(len(manifest)-counts['passed']),'tasks_with_process_failure_both_validators':sum(len(v)==2 and all(c=='process_requirement_missed' for c in v) for v in bytask.values()),'run_resources':[{'id':r['eval_run_id'],'score':r['validator_score'],'metadata':r.get('sandbox_metadata'),'inference_total':r.get('inference_total'),'inference_failed':r.get('inference_failure_count')} for r in runmeta],'causal_limit':'The category is the leading observed failure. Public race queries and private grading checks are unavailable. Source defects are reproduced with synthetic state, but the public projection does not prove which specific defect caused each failed episode.'}
(R/'diagnosis.json').write_text(json.dumps(summary,indent=2));(R/'procedure-control-probes.json').write_text(json.dumps({'scope':'Actual CHIDORI methods extracted and executed with synthetic state. No inference/environment calls to a service.','observations':cases},indent=2));print(json.dumps(summary,indent=2));print('CONTROL PROBES',json.dumps(cases,indent=2))
