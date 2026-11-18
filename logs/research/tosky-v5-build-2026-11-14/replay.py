"""Replay public observations with transport disabled. No new scores or shopper answers."""
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from agents.tests import test_tosky_v5 as helper
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'tosky-v4-local-review-2026-11-14/tasks'
FULL = 'local-c99b5b2fddef4178bd9903e1d0786a16'
PART = 'local-4efab7fafd3d4cc0be7418b3ceca0e54'

def snapshot(task, source, before_questions=False, run=FULL, stop_after=None):
    ns = helper.load(source)
    data = json.loads((DATA / run / (task + '.json')).read_text())
    ep = helper.previous.episode(ns)
    brief = copy.deepcopy(next(d['brief'] for d in data['diagnostics'] if d.get('kind') == 'brief'))
    ep.request, ep.brief, ep.roles = data['request'], brief, brief['roles']
    ep.items = [ns['Item'](x['name'], x.get('shelf', ''), x['quantity'], x['musts']) for x in brief['items']]
    ep.budget, ep.msg_limit = brief['budget'], brief['messages']
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
    return ns, ep, data

def audit_through_request(ns, ep, answer):
    before = copy.deepcopy(ep.brief)
    ep.items = []
    def respond(prompt, *args):
        if prompt.startswith(ns['BRIEF'][:60]):
            return copy.deepcopy(before)
        if prompt.startswith(ns['REQUEST_AUDIT'][:60]):
            return answer
        raise AssertionError('Unexpected model operation in request replay: ' + prompt[:80])
    ep.ask_json = respond
    ep.read_request()
    ep.owners_seen.clear()

def request_replay(source):
    out = {}
    cases = [
        ('t-559c052a99', {'kind':'purpose','property':'occasion','constraint':'wedding','quote':"they're for a wedding"}, 'p-904371f2b08b'),
        ('t-9fc67ee841', {'kind':'property','property':'pattern','constraint':'logo','quote':'with a logo pattern'}, 'p-00e4e97823f5'),
        ('t-b9f98ac0cc', {'kind':'property','property':'type of top','constraint':'tunik','quote':'the type of top is a tunik'}, 'p-02d14c7e9b88'),
    ]
    for task, addition, ident in cases:
        ns,e,d = snapshot(task, source)
        old = copy.deepcopy(e.brief['items'])
        audit_through_request(ns,e,{'add':[dict(item=1, **addition)]})
        assert ident in e.full
        need = addition['property'] + ': ' + addition['constraint']
        assert need in e.items[0].musts
        out[task] = {'before':old,'after':e.brief['items'],'recorded_choice_verdict':e.key_verdict(ident,need)}
    ns,e,d = snapshot('t-9ab5c575f5',source)
    old = e.owner_of('Jeans')
    audit_through_request(ns,e,{'alternatives':[{'item':1,'names':['pants','jeans'],'quote':'pants or jeans'}]})
    assert old == 0 and e.owner_of('Jeans') == 1
    out['t-9ab5c575f5']={'old_jeans_owner':old,'new_jeans_owner':e.owner_of('Jeans'),'items':e.brief['items']}
    return out

def interview_replay(source):
    out={}
    for task,run in [('t-82a9890683',PART),('t-70ef000b3f',FULL),('t-a6d85e34f3',FULL)]:
        ns,e,d=snapshot(task,source,True,run)
        plan=next(x['ask'] for x in d['diagnostics'] if x.get('kind')=='plan')
        e.ask_json=lambda *args:{'ask':plan}
        e.lanes=e.plan_lanes(3)
        e.lane_spent={1:0};e.lane_found={1:0};e.lane_used={1:set()}
        questions=[]
        for _ in range(3):
            n,labels=e.next_ask(e.lanes,e.lane_found,e.lane_spent,e.lane_used)
            if not labels:break
            questions.append(labels)
            e.settle(n,labels,False,[],False)
            e.lane_spent[n]+=1;e.msg_used+=1
        out[task]={'no_preference_branch_question_labels':questions,'lanes':e.lanes}
    assert 'options' not in [s for q in out['t-82a9890683']['no_preference_branch_question_labels'] for s in q]
    assert 'material' in [s for q in out['t-70ef000b3f']['no_preference_branch_question_labels'] for s in q]
    return out

if __name__ == '__main__':
    source=ROOT/'agents/Tosky_v5.py'
    result={'limitations':'Audit JSON is scripted from public request text. Interview output follows the no-new-preference branch only. Neither measures LLM extraction reliability, shopper responses, or scores.', 'request':request_replay(source),'interview':interview_replay(source)}
    (HERE/'public-replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
