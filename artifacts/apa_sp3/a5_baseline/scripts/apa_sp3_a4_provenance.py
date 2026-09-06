"""Per-kind import closures and a pinned, explicit r3-to-a4 compatibility bridge.

Prior art: dependency-directed invalidation, Make (Feldman 1979) and
content-addressed builds (Nix, Dolstra et al. 2004); unverified leads to check.
Reuse conservative dependency hashes; new: reviewed SP3 kind-specific bridge.
This is not a general semantic-equivalence prover. Unknown changes fail closed.
"""
from apa_sp3_common import ART, ROOT, REG_SHA, Red, read, sha
from apa_sp3_a4_registry import MANIFEST, CORRECTION, ORDER_SHA, manifest

BRIDGE = 'amendment_006_fingerprint.json'
FOLLOWUP = 'amendment_008_validation_followup.json'
ENGINE_KINDS = {'kernel', 'baseline', 'ppl', 'match', 'capture', 'decode',
                'margin', 'ceiling', 'ppl_long', 'capture_range'}
MODEL_KINDS = ENGINE_KINDS - {'margin'}


def closure(cell):
    kind = cell['kind']
    names = ['common.py', 'gpu.py', 'control.py', 'lead_gpu.sh', 'a4_provenance.py']
    paths = ['scripts/apa_sp3_' + n for n in names]
    paths += ['artifacts/apa_sp3/registration.json']
    if kind in ENGINE_KINDS:
        paths += ['artifacts/apa_sp3/build/manifest.json',
                  'artifacts/apa_sp3/adapter_import_cpu.json',
                  'artifacts/apa_sp3/weight_identity.json']
        paths += ['scripts/apa_sp3_'+n for n in
                  ('diag_bindings.cpp', 'make_diag.py', 'peak.cpp')]
    if kind in MODEL_KINDS:
        paths += ['scripts/apa_sp3_model.py']
    if kind in {'margin', 'calibration', 'match'}:
        paths += ['scripts/apa_sp3_metrics.py']
    if kind.startswith('capture_') or kind in {'ceiling', 'ppl_long', 'torch_reference'}:
        paths += ['scripts/apa_sp3_a4_registry.py', 'artifacts/apa_sp3/'+MANIFEST,
                  'artifacts/apa_sp3/'+CORRECTION, 'scripts/apa_sp3_a4_jobs.py']
    if kind in {'capture_range', 'capture_aggregate'}:
        paths += ['scripts/apa_sp3_a4_capture.py', 'scripts/apa_sp3_model.py']
    if kind in {'ppl_long', 'torch_reference'}:
        paths += ['scripts/apa_sp3_model.py', 'scripts/apa_sp3_a4_torch.py']
    if kind == 'torch_reference':
        paths += ['scripts/apa_sp3_a4_torch.py', 'artifacts/apa_sp3/weight_identity.json',
                  'artifacts/apa_sp3/a4_torch_sources.json']
    return sorted(set(paths))


def bridge():
    m = read(ART/BRIDGE)
    if (m['registration_sha256'] != REG_SHA or m['order_sha256'] != ORDER_SHA
            or sha(ART/BRIDGE) != (ART/(BRIDGE+'.sha256')).read_text().strip()
            or m['cell_amendment_sha256'] != sha(ART/MANIFEST)
            or m['correction_sha256'] != sha(ART/CORRECTION)):
        raise Red('fingerprint amendment binding changed')
    manifest()
    identity = sha(ART/BRIDGE)
    if (ART/FOLLOWUP).exists():
        follow = read(ART/FOLLOWUP)
        if (follow['registration_sha256'] != REG_SHA or follow['order_sha256'] != ORDER_SHA
                or follow['extends_sha256'] != identity
                or sha(ART/FOLLOWUP) != (ART/(FOLLOWUP+'.sha256')).read_text().strip()):
            raise Red('fingerprint followup binding changed')
        for path, change in follow['file_transitions'].items():
            previous = m['file_deltas'].get(path)
            if previous is None or previous['after_sha256'] != change['from_sha256']:
                raise Red('unknown fingerprint transition endpoint')
            previous['after_sha256'] = change['to_sha256']
        import hashlib
        identity = hashlib.sha256((identity+sha(ART/FOLLOWUP)).encode()).hexdigest()
    m['effective_sha256'] = identity
    return m


def current_fingerprint(cell):
    # Missing closure files are an execution blocker; never omit them.
    return {p: sha(ROOT/p) for p in closure(cell)}


def compatible(j, *, current=None, amendment=None):
    cell = j.get('cell', {})
    if not cell.get('kind') or j.get('registration_sha256') != REG_SHA:
        return False
    m = bridge() if amendment is None else amendment
    kind = cell['kind']
    try:
        now = current_fingerprint(cell) if current is None else current
    except FileNotFoundError:
        return False
    old = j.get('fingerprint', {})
    if j.get('fingerprint_schema') == 'apa_sp3_per_kind_v1':
        return old == now and j.get('fingerprint_amendment_sha256') == m['effective_sha256']
    # Only pinned legacy endpoints can cross this explicit bridge. A future
    # change to even one closure file rejects the old receipt automatically.
    for p, digest in now.items():
        if old.get(p) == digest:
            continue
        transition = m['file_deltas'].get(p, {})
        if not (old.get(p) in transition.get('before_sha256', [])
                and digest == transition.get('after_sha256')
                and kind in transition.get('unchanged_kinds', [])):
            return False
    return True
