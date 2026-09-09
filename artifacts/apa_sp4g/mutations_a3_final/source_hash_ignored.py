"""A3 isolated provenance. Prior art: SP3 (2026), Make/Feldman (1979),
SHA256/NIST (2001). Exact hash checks; no historical compatibility waiver.
"""
from apa_sp4g_common import *
from apa_sp4g_a3_registry import by_id, REGISTRATION, REGISTRATION_SHA

def preserved():
    before = read(A / 'a3_before.json')
    if sha(REGISTRATION) != REGISTRATION_SHA:
        raise Red('A3_REGISTRATION_CHANGED')
    reg = read(REGISTRATION)
    if sha(A / 'a3_before.json') != reg['before_sha256']:
        raise Red('A3_BEFORE_CHANGED')
    if sha(R / reg['order']['path']) != reg['order']['sha256']:
        raise Red('A3_ORDER_CHANGED')
    for section in ('source_sha256', 'receipt_sha256'):
        for p, h in before[section].items():
            if sha(R / p) != h:
                raise Red('A3_HISTORICAL_CHANGED: ' + p)
    verify_sources()
    build_check()
    return before

def fingerprint(c):
    paths = sorted(R.glob('scripts/apa_sp4g_a3_*.py'))
    paths += [R / 'scripts/apa_sp4g_a3_lead_gpu.sh', REGISTRATION,
              A / 'a3_before.json', A / 'A3_FORK_MERGE_AUDIT.md',
              A / 'registration.json', BUILD / 'manifest.json']
    # Imports are frozen in a3_before; also expose them directly in receipts.
    paths += [R / p for p in read(A / 'a3_before.json')['source_sha256']
              if p.startswith('scripts/')]
    return {str(p.relative_to(R)): sha(p) for p in paths}

def path_a3(name):
    return A / 'jobs_a3' / (name + '.json') if name in by_id() else job_path(name)

def require_a3(name):
    if name not in by_id():
        return require_pass(name)
    c = by_id()[name]
    j = read(path_a3(name))
    if (j.get('status') != 'PASS' or j.get('cell') != c
            or j.get('registration_sha256') != REG_SHA
            or j.get('a3_registration_sha256') != REGISTRATION_SHA
            or False):
        raise Red('A3_STALE_OR_RED_RECEIPT: ' + name)
    for d in c['depends']:
        require_a3(d)
    if j.get('dependencies') != {d: sha(path_a3(d)) for d in c['depends']}:
        raise Red('A3_DEPENDENCY_CHANGED')
    # Captures are small: verify bytes, not only inode/stat identity.
    for f in j.get('result', {}).get('files', []):
        if sha(R / f['path']) != f['sha256']:
            raise Red('A3_PAYLOAD_CHANGED: ' + f['path'])
    return j
