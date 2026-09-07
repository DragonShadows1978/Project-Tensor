"""A6 source/receipt audit. Prior art: A5 exact hash audit (2026), reused.
Separate source compatibility from local runtime identity; no identity waiver.
"""
import json
from apa_sp3_common import ART,ROOT,read,sha,require_pass,receipt_validation
from apa_sp3_a4_provenance import compatible,current_fingerprint,bridge


def audit():
    before=read(ART/'a6_before.json');amendment=bridge();rows=[];fingerprints={}
    for p,digest in before['receipts'].items():
        j=read(ROOT/p);kind=j.get('cell',{}).get('kind')
        if kind is not None and kind not in fingerprints:
            fingerprints[kind]=current_fingerprint(j['cell'])
        source={p:h for p,h in fingerprints.get(kind,{}).items() if p!='artifacts/apa_sp3/build/manifest.json'}
        copy=dict(j,fingerprint={p:h for p,h in j.get('fingerprint',{}).items() if p!='artifacts/apa_sp3/build/manifest.json'})
        row=dict(job=j['job'],kind=kind,receipt_status=j['status'],bytes_unchanged=sha(ROOT/p)==digest,
                 source_eligible=j['status']=='PASS' and compatible(copy,current=source,amendment=amendment))
        try:require_pass(j['job']);row['runtime_valid']=True
        except Exception as e:row.update(runtime_valid=False,runtime_reason=str(e))
        rows.append(row)
    result=dict(evidence_class='source and immutable receipt hash audit; no GPU validation',rows=rows,
                original_receipts=len(rows),original_bytes_unchanged=all(r['bytes_unchanged'] for r in rows),
                source_eligible=sum(r['source_eligible'] for r in rows),
                source_invalidated=[r['job'] for r in rows if r['receipt_status']=='PASS' and not r['source_eligible']],
                original_RED=[r['job'] for r in rows if r['receipt_status']=='RED'],
                runtime_valid=sum(r['runtime_valid'] for r in rows),
                note='Fresh seat had no build. Local host build identity differs from lead receipts; preserve lead build on integration. Source comparison excludes build key on BOTH sides; runtime validation never waives it.')
    (ART/'a6_receipt_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    with receipt_validation():r=audit()
    print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
