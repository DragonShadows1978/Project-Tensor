#!/usr/bin/env python3
"""APA-SP3 provenance and create-only receipts; no GPU imports at module scope.

Prior art: immutable experimental registration and content hashes are standard
reproducible-research practice; no novel data structure claimed. SP3 wiring is new.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_sp3'
BUILD = ART / 'build'
REG_SHA = 'd9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c'


class Red(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details=details


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, data):
    """Atomic create-only final JSON; never replace a result from a prior job."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.partial')
    with tmp.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)  # same filesystem; fails if final already exists
    finally:
        tmp.unlink()


def registration():
    if sha(ART / 'registration.json') != REG_SHA:
        raise Red('registration changed')
    return read(ART / 'registration.json')


def verify_sources(external=True):
    r = registration()
    for p, digest in r['source_sha256'].items():
        if sha(ROOT / p) != digest:
            raise Red(f'pre-existing source changed: {p}')
    if external:
        for p, digest in r['external_sha256'].items():
            if sha(p) != digest:
                raise Red(f'read-only source changed: {p}')
    return r


def protocol():
    """No CLI bypass: lead must recover and pin the original protocol first."""
    p = ART / 'protocol_amendment.json'
    if not p.exists():
        raise Red('G0 BLOCKED_PROTOCOL: original guide tokens/scoring source missing; token SHA null')
    j = read(p)
    if j.get('registration_sha256') != REG_SHA or j.get('status') != 'RECOVERED_ORIGINAL':
        raise Red('protocol amendment is not bound to this registration and original source')
    for key in ('source_path', 'tokens_path', 'scoring_source_path'):
        if sha(j[key]) != j[key.replace('_path', '_sha256')]:
            raise Red(f'protocol pin failed: {key}')
    if j['scoring'] != 'last_512_targets_within_input':
        raise Red('recovered scoring differs: a separately reviewed implementation amendment is required')
    import numpy as np
    ids = np.load(j['tokens_path'], allow_pickle=False)
    if ids.dtype != np.dtype('<i8') or ids.ndim != 1 or len(ids) < 32800:
        raise Red('need a 1D little-endian int64 original token stream with >=32800 tokens')
    if np.any(ids < 0) or np.any(ids >= 73448):
        raise Red('token ID out of model vocabulary')
    if hashlib.sha256(ids.tobytes()).hexdigest() != j['token_sha256']:
        raise Red('canonical token SHA mismatch')
    return j, ids


def seal_build():
    verify_sources()
    files = list(BUILD.glob('*.so')) + [BUILD / 'diagnostics.cu']
    m = read(BUILD / 'manifest.json')
    m['modules'] = {p.name: sha(p) for p in files}
    m['diagnostic_sources'] = {str(p.relative_to(ROOT)): sha(p) for p in
                               ROOT.glob('scripts/apa_sp3_diag*')}
    m['diagnostic_sources']['scripts/apa_sp3_make_diag.py'] = sha(ROOT / 'scripts/apa_sp3_make_diag.py')
    m['diagnostic_sources']['scripts/apa_sp3_peak.cpp'] = sha(ROOT / 'scripts/apa_sp3_peak.cpp')
    m['registration_sha256'] = REG_SHA
    (BUILD / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')


def load_runtime():
    verify_sources()
    m = read(BUILD / 'manifest.json')
    if m.get('registration_sha256') != REG_SHA:
        raise Red('unsealed build')
    for p, digest in m['sources'].items():
        if sha(ROOT / p) != digest:
            raise Red(f'stale build: {p}')
    for p, digest in m['modules'].items():
        if sha(BUILD / p) != digest:
            raise Red(f'changed module: {p}')
    for p, digest in m['diagnostic_sources'].items():
        if sha(ROOT / p) != digest:
            raise Red(f'stale diagnostic build: {p}')
    sys.path[:0] = [str(BUILD), str(ROOT / 'tensor_cuda')]
    import tensor_cuda as tc
    if Path(tc.__file__).resolve().parent != ROOT / 'tensor_cuda/tensor_cuda':
        raise Red('wrong Python engine checkout')
    if Path(tc._C.__file__).resolve().parent != BUILD:
        raise Red('wrong compiled engine checkout')
    return tc


def fingerprint():
    files = sorted(ROOT.glob('scripts/apa_sp3_*'))
    files += [BUILD / 'manifest.json', ART / 'registration.json']
    files += list(ART.glob('protocol_amendment.json'))
    return {str(p.relative_to(ROOT)): sha(p) for p in files if p.is_file()}


def require_pass(job):
    p = ART / 'jobs' / (job + '.json')
    if not p.exists():
        raise Red(f'BLOCKED_DEPENDENCY: {job}')
    j = read(p)
    if j.get('status') != 'PASS' or j.get('fingerprint') != fingerprint():
        raise Red(f'RED_OR_STALE_DEPENDENCY: {job}')
    for d,digest in j.get('dependencies',{}).items():
        if sha(ART/'jobs'/(d+'.json')) != digest:
            raise Red(f'changed dependency receipt: {d}')
    return j


def upward_float32(x):
    # Prior art: directed rounding, standard interval-arithmetic technique.
    import numpy as np
    if not math.isfinite(x) or x < 0:
        raise Red('margin must be finite nonnegative')
    f = np.float32(x)
    return float(np.nextafter(f, np.float32(-np.inf))) if float(f) < x else float(f)


if __name__ == '__main__':
    if sys.argv[1:] == ['seal-build']:
        seal_build()
    elif sys.argv[1:] == ['pins']:
        r = verify_sources()
        print(json.dumps({'status': 'PASS', 'evidence_class': 'code inspection',
                          'source_files': len(r['source_sha256']),
                          'kernel_bodies': sum(map(len, r['kernel_body_pins'].values()))}))
    else:
        raise SystemExit('usage: apa_sp3_common.py seal-build|pins')
