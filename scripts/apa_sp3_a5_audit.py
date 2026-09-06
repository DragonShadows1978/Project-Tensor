"""A5 receipt audit, including a4 kinds; no runtime identity waiver.

Prior art: a4 dependency-directed source audit. Reuse conservative per-kind
hash verification, separate source compatibility from actual runtime validity.
"""
import json
from apa_sp3_common import ART, ROOT, REG_SHA, read, sha, require_pass, receipt_validation
from apa_sp3_a4_provenance import bridge, closure, compatible


def audit():
    before = read(ART/'a5_before.json')
    amendment = bridge()
    rows = []
    for folder in ('jobs', 'jobs_a4'):
        for path in sorted((ART/folder).glob('*.json')):
            r = read(path)
            c = r['cell']
            current = {p:sha(ROOT/p) for p in closure(c)}
            source = {p:v for p,v in current.items() if p != 'artifacts/apa_sp3/build/manifest.json'}
            # Source-only projection removes the runtime-build key on BOTH
            # sides for exact per-kind receipts, including existing a4 jobs.
            source_receipt = dict(r, fingerprint={p:v for p,v in r.get('fingerprint',{}).items()
                                  if p != 'artifacts/apa_sp3/build/manifest.json'})
            row = dict(job=r['job'], kind=c['kind'], receipt_status=r['status'],
                       receipt_sha256=sha(path),
                       original_bytes_unchanged=before['receipts'].get(str(path.relative_to(ROOT))) in (None, sha(path)),
                       source_eligible=r['status']=='PASS' and compatible(source_receipt,current=source,amendment=amendment))
            try:
                require_pass(r['job'])
                row.update(runtime_valid=True)
            except Exception as e:
                row.update(runtime_valid=False, runtime_reason=str(e))
            rows.append(row)
    simulated = []
    for kind, paths in before['closures'].items():
        # Synthetic a4 receipt for every previously registered kind, including
        # unrun a4 kinds. Proves bridge behavior only, never GPU execution.
        fingerprint = {p:before['files'].get(p, sha(ROOT/p)) for p in paths}
        receipt = dict(cell=dict(kind=kind), registration_sha256=REG_SHA,
                       fingerprint_schema='apa_sp3_per_kind_v1', fingerprint=fingerprint,
                       fingerprint_amendment_sha256=before['effective_a4_sha256'])
        simulated.append(dict(kind=kind, compatible=compatible(receipt, amendment=amendment)))
    result = dict(evidence_class='source/receipt audit and synthetic a4 per-kind identity checks; no GPU work',
                  source_eligible=[r['job'] for r in rows if r['source_eligible']],
                  source_invalidated=[r['job'] for r in rows if r['receipt_status']=='PASS' and not r['source_eligible']],
                  runtime_valid=[r['job'] for r in rows if r['runtime_valid']],
                  original_RED=[r['job'] for r in rows if r['receipt_status']=='RED'],
                  original_receipts_unchanged=all(sha(ROOT/p)==digest for p,digest in before['receipts'].items()),
                  synthetic_a4_kind_checks=simulated, rows=rows,
                  note='A4-created receipt namespace audited too. Local build identity drift is not waived; source eligibility excludes only build/manifest.json. Lead retains original build; no rebuild or receipt rewrite authorized.')
    (ART/'a5_receipt_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    with receipt_validation():
        result=audit()
    print(json.dumps({k:len(result[k]) for k in ('source_eligible','source_invalidated','runtime_valid','original_RED')}))
