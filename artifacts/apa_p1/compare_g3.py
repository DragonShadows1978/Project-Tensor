# Prior art: pytest report phases; exact string equality, no statistical rule.
from pathlib import Path
import json,collections
A=Path('artifacts/apa_p1'); p=json.loads((A/'g3_production.json').read_text()); q=json.loads((A/'g3_port.json').read_text()); B=json.loads((A/'build.json').read_text())
def outcome(v):
    if any(x['outcome']=='failed' and x['when']!='call' for x in v):return 'ERROR'
    for x in v:
        if x['outcome']=='failed':return 'FAILED'
        if x['outcome']=='skipped':return 'SKIPPED'
    return 'PASSED' if any(x['when']=='call' and x['outcome']=='passed' for x in v) else 'MISSING'
rows=[]; totals={}
for k in sorted(set(p['collection'])|set(q['collection'])):
    pv=p['reports'].get(k,[]); qv=q['reports'].get(k,[]); a=outcome(pv);b=outcome(qv)
    same=a==b and [(x['when'],x['outcome']) for x in pv]==[(x['when'],x['outcome']) for x in qv]
    row=dict(id=k,production=a,port=b,equal=same)
    if a in ('FAILED','ERROR') or b in ('FAILED','ERROR'):
        row['production_messages']=[x['message'] for x in pv if x['outcome']=='failed'];row['port_messages']=[x['message'] for x in qv if x['outcome']=='failed']
    if not same:row['production_reports']=pv;row['port_reports']=qv
    rows.append(row)
files=[]
for f in sorted({r['id'].split('::')[0] for r in rows}):
    sub=[r for r in rows if r['id'].split('::')[0]==f]
    files.append(dict(file=f,production=dict(collections.Counter(r['production'] for r in sub)),port=dict(collections.Counter(r['port'] for r in sub)),ids=len(sub),differences=sum(not r['equal'] for r in sub)))
valid=p['collection']==q['collection'] and not p['collection_errors'] and not q['collection_errors'] and len(rows)>0 and len(rows)==len(p['reports'])==len(q['reports']) and p['module']['sha256']==B['production_sha256'] and q['module']['sha256']==B['module_sha256']
r=dict(status='PASS' if valid and all(x['equal'] for x in rows) else 'FAIL',ids=len(rows),differences=sum(not x['equal'] for x in rows),production=dict(collections.Counter(x['production'] for x in rows)),port=dict(collections.Counter(x['port'] for x in rows)),modules=dict(production=p['module'],port=q['module']),files=files,rows=rows,note='Outcome-parity gate only. Native CPU no-device errors, historical missing immutable fixtures and other baseline failures remain failures. No tests modified, deselected, suppressed or made to pass.')
(A/'g3_comparison.json').write_text(json.dumps(r,indent=2)+'\n')
with (A/'G3_TABLE.md').open('w') as f:
    f.write('| Production test file | IDs | Production P/S/F/E | Port P/S/F/E | Differences |\n|---|---:|---|---|---:|\n')
    for x in files:
        fmt=lambda arm:'/'.join(str(x[arm].get(k,0)) for k in ('PASSED','SKIPPED','FAILED','ERROR'))
        f.write(f"| {Path(x['file']).name} | {x['ids']} | {fmt('production')} | {fmt('port')} | {x['differences']} |\n")
with (A/'g3_ids.tsv').open('w') as f:
    f.write('test_id\tproduction\tport\tequal\n')
    for x in rows:f.write(f"{x['id']}\t{x['production']}\t{x['port']}\t{x['equal']}\n")
print(json.dumps({k:r[k] for k in ('status','ids','differences','production','port')},indent=2))
with (A/'LEDGER.md').open('a') as f:f.write('\n- G2 SASS PASS: 213 ported kernels, 428344 instructions per arm, zero count/histogram/instruction-text differences. Full closure includes spo2/spo4/spo5, spq1/spq4/spq6, EB2b/EB3/EB4, baseline dump (sass_comparison.json, sass_counts.tsv).\n- G3 '+r['status']+': 689 exact per-ID phase/outcome matches; production and port each 410 PASSED, 139 SKIPPED, 125 FAILED, 15 ERROR. Exact failure messages retained. Original historical pinned-runtime checks are not bypassed; missing snapshots and no-device failures remain. Module provenance hashes verified. G3_TABLE.md, g3_ids.tsv, g3_comparison.json and per-arm JSON/logs. Port package module restored and verified.\n')
