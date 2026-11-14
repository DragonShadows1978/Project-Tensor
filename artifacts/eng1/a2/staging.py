#!/usr/bin/env python3
"""Use unchanged stage.py; publish a distinct A2 seal without touching old seals.
Prior art: ENG1 stage.py / GN1 A8 seals (2026), SHA256 NIST FIPS 180-4 (2015).
Taken: all original census/arm/seal checks; new: output seal routing and A2
order/adapter registration. No engine/package bytes or numerical gates change.
"""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

A = Path(__file__).resolve().parent
R = A.parents[2]
T = R / 'artifacts/eng1/tool'
sys.path.insert(0, str(T))
S = importlib.import_module('stage')
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
mode = sys.argv[1]
if mode in ('arms', 'census'):
    getattr(S, 'step_' + mode)()
elif mode == 'seal':
    draft = Path('/tmp/eng1_a2_work/staging_seal_draft.json')
    S.SEAL_NAME = str(draft)
    S.step_seal()  # retains all original checks, including staged test equality
    data = json.loads(draft.read_text())
    data['order_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True,
                                         capture_output=True, text=True).stdout.strip()
    data['order_sha256'] = S.sha(R / 'orders/ENG1_A2_REV2_FIXES.md')
    data['base_commit'] = '7cf46bd5'
    data['a2_note'] = 'A2 order; original stage.py seal checks; unchanged tools, new seal routing in a2/sitecustomize.py'
    for p in sorted(A.iterdir()):
        if p.is_file() and (p.suffix == '.py' or p.name == 'REGISTRATION.md'):
            data['registration_files'][str(p.relative_to(R))] = S.sha(p)
    dest = T / 'staging_seal_a2.json'
    with dest.open('x') as f:
        f.write(json.dumps(data, indent=1, sort_keys=True) + '\n')
    print('new seal:', dest.relative_to(R), 'sha256:', S.sha(dest))
else:
    raise ValueError(mode)
