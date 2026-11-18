"""Offline prompt-format audit and replay; all inference transport is stubbed."""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from agents.tests.test_tosky_l17 import load, episode
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'agents/Tosky_l17.py'
BEFORE=Path('/tmp/Tosky_l17-before-format.py')
ns=load(); old=load(BEFORE) if BEFORE.exists() else None
ROLES=('BRIEF','NOTE','CHECK','AUDIT','UNMIX','TRAIL','SWAP','GATE','SHAPE','FINISH_ARGS','RANK_PROMPT','HEAR_PROMPT')

def methods(tree):
    return {c.name+'.'+m.name:ast.dump(m,include_attributes=False)
            for c in tree.body if isinstance(c,ast.ClassDef)
            for m in c.body if isinstance(m,ast.FunctionDef)}

source=SOURCE.read_text(); tree=ast.parse(source)
for name in ROLES+('GUIDE','READING','CONTEXT_RULES'):
    ET.fromstring('<root>'+ns[name]+'</root>')
for text in ns['GUIDE_MODULES'].values():ET.fromstring('<root>'+text+'</root>')
rate_methods=('call_model','quote_inference','think','judges','cache_messages','token_prefix','offered','msgs_free','act','take_heard','run_calls','finish')
preserved={}
if old:
    previous=methods(ast.parse(BEFORE.read_text())); current=methods(tree)
    preserved={name:previous['Cairn.'+name]==current['Cairn.'+name] for name in rate_methods}
    assert all(preserved.values())

for node in ast.walk(tree):
    if isinstance(node,(ast.Import,ast.ImportFrom)):
        names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
        assert not {'os','socket','http','urllib','requests','subprocess','shutil','ctypes','cffi','pickle','base64','binascii','codecs','zlib'} & {name.split('.')[0] for name in names}
    if isinstance(node,ast.Call):
        name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
        assert name not in {'exec','eval','__import__','open','write_text','write_bytes','unlink','mkdir'}
assert SOURCE.stat().st_size<1_000_000

# Capture the exact messages given to the proxy method without sending anything.
ns,e=episode(); bodies=[]
def post(path,body):
    assert path=='/inference/chat/completions'
    bodies.append(body)
    return SimpleNamespace(data={'choices':[{'message':{'content':'{}'}}],'usage':{'cost':0}},error=None)
ns['_llm_within']=lambda _:SimpleNamespace(post_verbose=post)
examples=[('HEAR',ns['HEAR_PROMPT'],{'question':'Which finish do you require?',
            'shopper_reply':'I need matte, and keep the total under 30.', 'product_names':'sample item'}),
          ('BRIEF',ns['BRIEF'],{'shopper_request':'Buy one sample item with a matte finish. Total budget 30.',
            'available_tools':e.tools})]
preview=['Rendered prompt examples from the offline proxy stub.',
         'Fictional inputs; no paid inference was sent. Cache metadata is omitted below.', '']
for name,prompt,data in examples:
    e.ask_json((prompt,data),[ns['LIGHT']],100)
    preview.append('=== '+name+' ===')
    for message in bodies[-1]['messages']:
        text=message['content']
        if isinstance(text,list):text='\n'.join(block.get('text','') for block in text)
        ET.fromstring('<root>'+text+'</root>')
        preview.extend(['',message['role'].upper(),text,''])
(OUT/'prompt-preview.txt').write_text('\n'.join(preview))

results=[];started=time.monotonic()
for path in sorted((ROOT/'logs/research/miyagi-2026-11-17/feedback').glob('*.json')):
    feedback=json.loads(path.read_text())
    value={'source':'store_observations','events':[
        {'source':call.get('tool_name'),'step':turn.get('step_index'),'action':call.get('args'),
         'observation':call.get('observation'),'done':call.get('done'),'error_code':call.get('error_code')}
        for turn in feedback.get('interaction') or [] for call in turn.get('calls') or []]}
    rendered=ns['_source_block']('store_observations',value)
    payload=json.loads(ET.fromstring(rendered).text)
    restored=ns['_unpack_context'](payload) if payload.get('format')=='cairn-context-v2' else payload
    assert ns['_dump'](restored)==ns['_dump'](value)
    result={'feedback_file':path.name,'raw_bytes':len(ns['_dump'](value).encode()),
            'readable_block_bytes':len(rendered.encode()),'exact_round_trip':True}
    if old:result['previous_packed_bytes']=len(old['_context_text'](value).encode())
    results.append(result)
raw=sum(x['raw_bytes'] for x in results); new=sum(x['readable_block_bytes'] for x in results)
measure={'episodes':len(results),'raw_json_bytes':raw,'readable_xml_block_bytes':new,
         'reduction_against_raw_json':1-new/raw,'all_round_trips_exact':True,
         'elapsed_seconds':time.monotonic()-started,
         'scope':'Whole-episode released observation blocks; not a live token, cost or accuracy measurement.'}
if old:
    previous=sum(x['previous_packed_bytes'] for x in results)
    measure.update({'previous_packed_json_bytes':previous,'size_increase_against_previous_codec':new/previous-1})
(OUT/'packing-measurements.json').write_text(json.dumps({'summary':measure,'episodes':results},indent=2)+'\n')
report=json.loads((OUT/'verification.json').read_text())
report.update({'after_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'after_bytes':SOURCE.stat().st_size,
    'static_roles_checked':list(ROLES),'xml_fragments_valid':True,'preserved_controller_methods':preserved,
    'source_security_check':True,'paid_calls':0,'preview':'prompt-preview.txt','measurements':measure,
    'remote_admission_run':False})
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'roles':len(ROLES),'preserved_controller_methods':len(preserved),'measurements':measure,
                  'source_bytes':SOURCE.stat().st_size,'paid_calls':0},indent=2))
