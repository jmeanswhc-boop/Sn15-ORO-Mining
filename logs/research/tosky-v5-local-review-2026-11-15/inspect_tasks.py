import json
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
ALIASES={'v5':'local-6dfac68554764b9e8f2f025d9f0a1777','v4':'local-c99b5b2fddef4178bd9903e1d0786a16','v4a':'local-4efab7fafd3d4cc0be7418b3ceca0e54','v3':'local-cb574c03f70841abb0efec38dba1f2f0'}
run=ALIASES[sys.argv[1]]
for task in sys.argv[2:]:
 d=json.loads((BASE/'tasks'/run/(task+'.json')).read_text())
 print('\nCASE',task,run,d['score'],d['category'],d['terminal'])
 print('REQUEST',d['request'])
 for row in d['diagnostics']:
  if row.get('kind') in ['brief','request_audit','plan','ask','hear','question_coverage','quiet_reply','list_settled','lists_off','line_hold','terminal_hold','key_hold','cap_cart','end','budget_denied','llm_fail','swap','stockout_scope','audit','decline_ok','forced_buy','late_decline_held','need_search']:
   print(json.dumps(row,ensure_ascii=False))
 print('ACTIONS')
 for e in d['world']:
  if e.get('kind') in ['action','message','reply','event','finish']:
   print(json.dumps(e,ensure_ascii=False)[:1100])
 print('MODEL OUTPUTS')
 for c in d['calls']:
  if c['purpose'] in ['REQUEST_AUDIT','HEAR','PLAN','CHECK','other'] or c['number']>=len(d['calls'])-3:
   print(c['number'],c['purpose'],c['model'],c['finish_reason'],json.dumps(c['content'],ensure_ascii=False)[:1600],json.dumps(c['tool_calls'],ensure_ascii=False)[:700])
