"""A2 isolation/provenance. Prior art: SP3 (2026), Make/Feldman1979,
SHA256/NIST2001. Historical closures remain unchanged; no compatibility waiver.
"""
from apa_sp4g_common import *
from apa_sp4g_a2_registry import by_id

def paths():
    return sorted(R.glob('scripts/apa_sp4g_a2_*.py'))+[R/'scripts/apa_sp4g_a2_lead_gpu.sh']

def fingerprint_a2(c):
    # Full A2 driver plus unchanged A1 execution closure; new receipts only.
    p=paths()+[R/f'scripts/apa_sp4g_{n}.py' for n in ('common','model','metrics','gpu','registry','a1_provenance')]
    p += [A/'registration.json',BUILD/'manifest.json',A/'amendment_006_a2_hypotheses.json',A/'amendment_007_a2_execution.json']
    return {str(x.relative_to(R)):sha(x) for x in p}

def path_a2(name):return A/'jobs_a2'/f'{name}.json' if name in by_id() else job_path(name)

def require_a2(name):
    if name not in by_id():return require_pass(name)
    j=read(path_a2(name));c=by_id()[name]
    if j.get('status')!='PASS' or j.get('cell')!=c or j.get('registration_sha256')!=REG_SHA or j.get('fingerprint')!=fingerprint_a2(c):raise Red('A2_STALE_OR_RED_RECEIPT: '+name)
    for d in c['depends']:require_a2(d)
    if j.get('dependencies')!={d:sha(path_a2(d)) for d in c['depends']}:raise Red('A2_DEPENDENCY_CHANGED')
    for f in j.get('result',{}).get('files',[]):
        if 'stat' in f:
            st=(R/f['path']).stat()
            if {k:getattr(st,k) for k in f['stat']}!=f['stat']:raise Red('A2_PAYLOAD_STAT_CHANGED')
        elif sha(R/f['path'])!=f['sha256']:raise Red('A2_PAYLOAD_CHANGED')
    return j

def preserved():
    before=read(A/'a2_before.json')
    for p,h in before['receipt_sha256'].items():
        if sha(R/p)!=h:raise Red('A2_HISTORICAL_RECEIPT_CHANGED: '+p)
    for p,h in before['source_sha256'].items():
        if p.endswith(('RESULTS.md','cells.json','lead_commands.txt','apa_sp4g_report.py')):continue
        if sha(R/p)!=h:raise Red('A2_HISTORICAL_EXECUTION_CHANGED: '+p)
    verify_sources();build_check()
    return before
