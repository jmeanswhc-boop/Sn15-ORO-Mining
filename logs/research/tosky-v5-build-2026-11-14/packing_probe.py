import json
import runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent
h=runpy.run_path(str(HERE/'replay.py'))

def pack(records):
    groups={}
    def leaves(value,path=()):
        if isinstance(value,dict) and value:
            return [leaf for key in sorted(value) for leaf in leaves(value[key],path+(key,))]
        return [(path,value)]
    for record in records:
        flat=leaves(record);columns=tuple(path for path,_ in flat)
        groups.setdefault(columns,[]).append([value for _,value in flat])
    return json.dumps({'tables':[{'columns':cols,'rows':rows} for cols,rows in groups.items()]},ensure_ascii=False,separators=(',',':'))

if __name__=='__main__':
    rows=[]
    for task in ['t-559c052a99','t-abf1c45838','t-82a9890683','t-13e82f2c08','t-9fc67ee841','t-9ab5c575f5']:
        ns,e,d=h['snapshot'](task,Path('agents/Tosky_v5.py'))
        ids=list(reversed(e.seen_order))[:25]
        raw=[e.full.get(i) or e.rows[i] for i in ids]
        old='\n'.join(ns['_dump'](ns['_clip'](r,200))[:1200] for r in raw)
        compact=pack(raw)
        rows.append({'task':task,'records':len(raw),'old_clipped_chars':len(old),'lossless_table_chars':len(compact),'raw_chars':len(ns['_dump'](raw))})
    (HERE/'05b-packing-feasibility.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
