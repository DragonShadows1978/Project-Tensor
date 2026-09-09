"""Exact A1 compatibility bridge, following SP3 a4 (2026).

Prior art: SP3 a4, Make/Feldman (1979) dependency hashes. Only an explicitly
reviewed kernel endpoint is reusable; no semantic-equivalence algorithm.
"""
from apa_sp4g_common import A,R,REG_SHA,Red,read,sha,fingerprint

BRIDGE='amendment_003_a1_fingerprint.json'

def bridge():
    p=A/BRIDGE;m=read(p)
    if (sha(p)!=(A/(BRIDGE+'.sha256')).read_text().strip()
            or m['registration_sha256']!=REG_SHA
            or m['execution_amendment_sha256']!=sha(A/'amendment_002_a1_execution.json')
            or m['order_sha256']!=sha(R/'orders/APA_SP4G_AMENDMENT_1.md')
            or m['before_sha256']!=sha(A/'a1_before.json')):
        raise Red('A1_FINGERPRINT_AMENDMENT_CHANGED')
    return dict(m,sha256=sha(p))

def legacy_compatible(j,*,current=None,amendment=None):
    if j.get('status')!='PASS' or j.get('cell',{}).get('id')!='kernel512':return False
    m=bridge() if amendment is None else amendment;k=m['legacy_kernel']
    now=fingerprint(j['cell']) if current is None else current
    return (j.get('registration_sha256')==REG_SHA
            and sha(A/'jobs/kernel512.json')==k['receipt_sha256']
            and j.get('fingerprint_schema')=='apa_sp4g_per_kind_v1'
            and j.get('cell')==k['cell'] and j.get('dependencies')=={}
            and j.get('fingerprint')==k['before_fingerprint']
            and now==k['after_fingerprint'])
