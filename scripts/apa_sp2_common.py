"""SP2 receipt and provenance helpers; no device work at import time."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_sp2'
NOTE = 'kernel sweep; this establishes nothing about model quality'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registration():
    assert sha(ART/'registration.json') == (ART/'registration.sha256').read_text().split()[0]
    r = json.loads((ART/'registration.json').read_text())
    for path, digest in r['parent_registrations'].items():
        assert sha(ROOT/path) == digest
    return r


def load_runtime():
    build = ART/'build'
    manifest = json.loads((build/'manifest.json').read_text())
    for path, digest in manifest['sources'].items():
        assert sha(ROOT/path) == digest, f'stale build source: {path}'
    for name, digest in manifest['modules'].items():
        assert sha(build/name) == digest, f'stale module: {name}'
    assert manifest['modules'], 'empty build manifest'
    sys.path[:0] = [str(build), str(ROOT/'tensor_cuda')]
    import tensor_cuda as tc
    assert Path(tc.__file__).resolve().parent == ROOT/'tensor_cuda/tensor_cuda'
    assert Path(tc._C.__file__).resolve().parent == build
    return tc


def fingerprint():
    paths = [ART/'registration.json', ART/'build/manifest.json',
             ROOT/'tensor_cuda/tensor_cuda/apa_sp2.py', ROOT/'tensor_cuda/tensor_cuda/quant.py',
             ROOT/'tensor_cuda/tests/apa_sp1_reference.py',
             ROOT/'tensor_cuda/tests/apa_sp1_1_reference.py']
    paths += sorted((ROOT/'scripts').glob('apa_sp2_*.py'))
    paths += [ROOT/'scripts/apa_sp2_lead_gpu.sh']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def receipt(target, data):
    data.update(target=target, registration_sha256=sha(ART/'registration.json'),
                fingerprint=fingerprint(), scope_note=NOTE)
    path = ART/'gpu'/f'{target}.{time.time_ns()}.json'
    write_new(path, data)
    print(path.relative_to(ROOT), data['status'], flush=True)
    return path


def attempts(target):
    fp = fingerprint()
    found = []
    for path in sorted((ART/'gpu').glob(f'{target}.*.json')):
        row = json.loads(path.read_text())
        if row.get('fingerprint') == fp:
            if row.get('status') == 'WORKER_EXIT':
                if row['returncode'] == 0:
                    continue
                row['status'] = 'ERROR'
            row['_receipt'] = str(path.relative_to(ROOT))
            found.append(row)
    return found


def latest(target):
    rows = attempts(target)
    return rows[-1] if rows else None


def targets(reg=None):
    reg = registration() if reg is None else reg
    measure = [f'eq_b{b}_{s["id"]}' for b in reg['bits'] for s in reg['shapes']]
    sweep = [f'sweep_b{b}_{s["id"]}_e{i}' for b in reg['bits'] for s in reg['shapes']
             for i in range(len(reg['epsilon_grid']))]
    return measure, sweep


def decode_target(target, reg=None):
    reg = registration() if reg is None else reg
    for b in reg['bits']:
        for i, s in enumerate(reg['shapes']):
            if target == f'eq_b{b}_{s["id"]}':
                return 'eq', b, s, i, None
            for e in range(len(reg['epsilon_grid'])):
                if target == f'sweep_b{b}_{s["id"]}_e{e}':
                    return 'sweep', b, s, i, e
    raise ValueError('unregistered target; use list')
