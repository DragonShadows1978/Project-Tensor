"""Read-only receipt audit: source compatibility versus local runtime readiness.

Prior art: dependency closure auditing, standard reproducible-build practice;
reuse exact content/stat hashes, new SP3 reporting only.
"""
import json
from apa_sp3_common import ART, ROOT, read, sha, require_pass, receipt_validation
from apa_sp3_a4_provenance import bridge, closure, compatible


def audit():
    amendment=bridge()
    rows=[]
    for p in sorted((ART/'jobs').glob('*.json')):
        r=read(p);cell=r.get('cell',{})
        row=dict(job=r['job'],kind=cell.get('kind'),receipt_status=r['status'],receipt_sha256=sha(p))
        if r['status']!='PASS':
            row.update(source_eligible=False,runtime_valid=False,reason='Original '+r['status']+' preserved; '+str(r.get('error')))
        else:
            # This projection asks only whether OUR SOURCE EDIT requires a
            # rerun. It explicitly does not waive runtime fingerprint changes.
            current={k:sha(ROOT/k) for k in closure(cell) if k!='artifacts/apa_sp3/build/manifest.json'}
            source_ok=compatible(r,current=current,amendment=amendment)
            row['source_eligible']=source_ok
            try:
                require_pass(r['job'])
                row.update(runtime_valid=True,reason='PASS: source and current runtime/dependencies match')
            except Exception as e:
                row.update(runtime_valid=False,reason=str(e))
        rows.append(row)
    eligible=[r['job'] for r in rows if r['source_eligible']]
    invalid=[r['job'] for r in rows if r['receipt_status']=='PASS' and not r['source_eligible']]
    result=dict(evidence_class='code/source and receipt audit; not GPU rerun',source_eligible=eligible,
                source_invalidated=invalid,runtime_valid=[r['job'] for r in rows if r['runtime_valid']],
                original_RED=[r['job'] for r in rows if r['receipt_status']=='RED'],rows=rows,
                note='Source eligibility isolates r3->a4 source edits. This seat rebuilt locally for CPU gates; keep lead original build/modules. Live runtime validation reruns read-only after merge; no receipt is rewritten.')
    (ART/'a4_receipt_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    with receipt_validation():r=audit()
    print(json.dumps({k:len(r[k]) for k in ('source_eligible','source_invalidated','runtime_valid','original_RED')}))
