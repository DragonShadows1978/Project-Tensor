"""Frozen STE5 evidence schema.

Prior art: NIST FIPS 180-4 (2015), SHA-256 provenance; IEEE 754-2019,
full-width representations. Unverified - lead to check those standards.
Exact gate/configuration rules are the lead's STE5 order (2026);
no prior art known to me for that particular experimental protocol.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / 'apa_ste1'
LEAD = HERE / 'lead'
KEYS = ('out', 'lse', 'thr', 'dq', 'dk', 'dv')
INPUTS = ('q', 'k', 'kq', 'v', 'do')
SHAPES = {'small1': (1, 2, 7, 11, 32, 16, 2),
          'small2': (1, 4, 17, 19, 32, 16, 2),
          'grapa': (1, 16, 4096, 4096, 128, 64, 16)}
CONFIGS = {'base': ('base', None, False), 'off': ('arm', 0, False),
           'unset': ('arm', None, False), 'on': ('arm', 1, False),
           'off-all': ('arm', 0, True), 'on-all': ('arm', 1, True)}
N = 3
ARM_PIN = 'af9ffce4733628caac5b6bd7a345de86c66c1f67d5150a53d7454b3d4bcb3760'
BASE_PIN = '69b154cd2e9a5a7cbe288e1419d963697e5add57d2b4ee827c4c78a273ff2c5c'
BASE_PACKAGE = Path('/mnt/ForgeRealm/engines/tf32-det-69b154cd2e9a/tensor_cuda/tensor_cuda')
ARM_PACKAGE = OLD / 'engine/tensor_cuda/tensor_cuda'
SO_NAME = '_tensor_cuda.cpython-312-x86_64-linux-gnu.so'
Z = float(np.float32(1.0364333894937898))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def cpu_environment():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU only'
    environment()


def environment():
    assert not sys.flags.optimize, 'assertions must remain enabled'
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert os.environ.get(k) == '1', k
    assert Path(os.environ['TMPDIR']).resolve() == HERE / 'scratch'


def label(config, suite):
    return f'ste5v1-{config}-{suite}'


def names(suite):
    assert suite in ('small', 'grapa')
    return ('small1', 'small2') if suite == 'small' else ('grapa',)


def folder(config, suite, run):
    return LEAD / label(config, suite) / f'run-{run}'


def bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def decode(x):
    assert x.dtype in (np.dtype('uint16'), np.dtype('uint32')), x.dtype
    if x.dtype == np.uint16:
        return (x.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return x.view(np.float32).astype(np.float64)


def healthy(x):
    x = np.asarray(x)
    assert np.isfinite(x).all() and np.any(x != 0) and np.any(x != x.flat[0]), 'nonfinite/zero/constant operand'


def fixture(name):
    return OLD / 'inputs' / f'{name}.npz'


def input_words(name):
    with np.load(fixture(name), allow_pickle=False) as d:
        return {k: bf16(d[k]) for k in INPUTS}


def check_pins():
    r = json.loads((HERE / 'INPUT_PINS.json').read_text())
    assert r['N'] == N and r['variants'] == ['g1', 'h']
    assert r['G-G5'] == {'cosine_min': .995, 'slope_min': .98, 'slope_max': 1.02, 'relative_L2_slack': .01}
    # Separate immutable amendments may replace harness pins, never thresholds.
    # Prior art: append-only provenance ledgers (STE4, 2026), reused.
    for amendment in sorted(HERE.glob('INPUT_PINS_AMENDMENT_*.json')):
        a=json.loads(amendment.read_text())
        assert a['input_pins_sha256']==digest(HERE/'INPUT_PINS.json')
        for p, row in a['overrides'].items():
            assert r['pins'][p]==row['before']
            r['pins'][p]=row['after']
    for p, h in r['pins'].items():
        assert digest(ROOT / p) == h, f'pinned artifact changed: {p}'
    assert digest(BASE_PACKAGE / SO_NAME) == BASE_PIN, 'base .so changed'
    assert digest(ARM_PACKAGE / SO_NAME) == ARM_PIN, 'arm .so changed'
    return r


def governor(path, labels):
    path = Path(path)
    rows = [s.split('\t') for s in path.read_text().splitlines()] if path.exists() else []
    result = {}
    for job in labels:
        hits = [r for r in rows if len(r) >= 3 and r[2] == job]
        ok = (len(hits) == 1 and len(hits[0]) == 9 and hits[0][-2:] == ['ran', '0']
              and Path(hits[0][1]).resolve() == ROOT)
        result[job] = {'PASS': bool(ok), 'rows': hits}
    return result


def require_governor(path, labels):
    result = governor(path, labels)
    missing = [k for k, v in result.items() if not v['PASS']]
    assert not missing, 'missing governor ran/rc0: ' + ', '.join(missing)
    return result


def import_old_driver():
    spec = importlib.util.spec_from_file_location('ste4_readonly_ste1', OLD / 'gpu_gate.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def require_premise():
    report=json.loads((LEAD/'PREMISE.json').read_text())
    assert report['PASS'] and report['premise_only'] and report['gates']=={'G-G1':True}
    assert report['input_pins_sha256']==digest(HERE/'INPUT_PINS.json')
    assert report['effective_pins_sha256']==evidence_pins_digest()
    assert report['verifier_sha256']==digest(HERE/'verify5.py')
    for row in report['base_inventory']:
        assert digest(folder(row['config'],row['suite'],row['run'])/'capture.json')==row['capture_sha256']
    require_governor(LEAD/'governor.tsv',[label('base',s) for s in ('small','grapa')])


def evidence_pins_digest():
    # Bind each capture to both original pins and explicit immutable amendments.
    parts=[digest(HERE/'INPUT_PINS.json')]+[digest(p) for p in sorted(HERE.glob('INPUT_PINS_AMENDMENT_*.json'))]
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
