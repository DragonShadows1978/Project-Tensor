"""PF2-F1 arms and CPU-safe support. Prior art: PF2 arm isolation (2026),
copied packages and fixed-width unsigned views; ordinary test bookkeeping.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_pf2f1'
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_pf2f1_build as build


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(data, f, indent=2, default=str); f.write('\n')


def arm_dir(label):
    return ROOT / 'artifacts/apa_p1pf2/arms/PF' if label == 'PF' else ART / 'arms' / label


def precheck(label):
    rec = build.receipt(label)
    arm = json.loads((arm_dir(label) / 'ARM.json').read_text())
    assert sha(ROOT / rec['module']) == rec['module_sha256']
    assert sha(ROOT / arm['module']) == arm['module_sha256'] == rec['module_sha256']
    for rel, h in arm['python_files'].items():
        assert sha(ROOT / rel) == h, ('python drift', rel)
    if label != 'PF':
        assert build.source_map(Path(rec['source_root'])) == rec['source_sha256_after']
    return dict(label=label, module_sha256=rec['module_sha256'])


def stage(label):
    assert label != 'PF'
    rec = build.receipt(label)
    dest = arm_dir(label) / 'tensor_cuda'
    dest.mkdir(parents=True)
    files = {}
    for p in sorted((ROOT / 'tensor_cuda/tensor_cuda').rglob('*.py')):
        if '__pycache__' in p.parts:
            continue
        q = dest / p.relative_to(ROOT / 'tensor_cuda/tensor_cuda')
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        files[str(q.relative_to(ROOT))] = sha(q)
    module = ROOT / rec['module']
    shutil.copyfile(module, dest / module.name)
    save(dest.parent / 'ARM.json', dict(label=label, module=str((dest / module.name).relative_to(ROOT)),
         module_sha256=sha(dest / module.name), python_files=files))
    print('STAGED', precheck(label))


def load_arm(label):
    precheck(label)
    dest = arm_dir(label)
    sys.path.insert(0, str(dest))
    import tensor_cuda as tc
    assert Path(tc.__file__).resolve().parent == (dest / 'tensor_cuda').resolve()
    assert sha(tc._C.__file__) == build.receipt(label)['module_sha256']
    return tc, dict(arm=label, module_sha256=sha(tc._C.__file__))


def torch_setup():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return torch


def informative(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    u = a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]).reshape(-1)
    assert u.size > 0 and (u != u[0]).any(), 'empty or constant comparison operand'
    return u


def bit_equal(a, b):
    import numpy as np
    assert a.shape == b.shape and a.dtype == b.dtype, 'shape/dtype mismatch before unsigned view'
    ua, ub = informative(a), informative(b)
    assert ua.dtype.kind == ub.dtype.kind == 'u' and ua.dtype.itemsize == a.dtype.itemsize
    return bool(np.array_equal(ua, ub))


# Read-only reference paths retained from PF2; no imported writer is invoked.
PACKED_TABLE = ROOT / 'artifacts/apa_p1d/packed_e_q_table.json'
DENSE_TABLE = Path('/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2/e_q_table.json')
DENSE_TABLE_SHA = '5b6f8bfb405b40c03da21ad8d50eb40c11b85ed9559ee717d2a105c90c0e618d'


if __name__ == '__main__':
    stage(sys.argv[1])
