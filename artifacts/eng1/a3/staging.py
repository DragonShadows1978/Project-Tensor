#!/usr/bin/env python3
"""Use unchanged stage.py; publish a distinct A3 seal without touching old seals (follows a2/staging.py).
Prior art: a2/staging.py and ENG1 stage.py / GN1 A8 seals (2026); SHA-256 NIST FIPS 180-4 (2015).
Taken: stage.py's step_seal with all its original checks (arm maps, engine pins, staged-test equality), the
draft-to-scratch-then-publish-with-open('x') pattern and the order/registration overrides from a2/staging.py.
New in A3: the A3 order and its commit; A2's registered adapter files carried over (each re-hashed and asserted equal
to the A2 seal); A3's two tool-path files registered; explicit equality checks that the A3 seal differs from the A2
seal only where route (a) changed bytes (tool_files['evidence.py']). No engine, arm, census or test bytes change.
The order commit was read from the worktree's ref file (refs/heads/eng1), not from a git command (the dispatch
forbids git); it equals the order's stated commit 46676057.
"""
import importlib
import json
import os
from pathlib import Path
import sys

A = Path(__file__).resolve().parent
R = A.parents[2]
T = R / 'artifacts/eng1/tool'
ORDER = 'orders/ENG1_A3_DEVICE_SEAL_ROUTING.md'
ORDER_COMMIT = '46676057f27fedcad214e214d20f5fd699f89fe3'
A3_REGISTERED = ('artifacts/eng1/a3/staging.py', 'artifacts/eng1/a3/sitecustomize.py')
sys.path.insert(0, str(T))
S = importlib.import_module('stage')
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert sys.argv[1:] == ['seal'], sys.argv
assert ORDER_COMMIT.startswith('46676057') and len(ORDER_COMMIT) == 40
draft = Path('/tmp/eng1_a3_work/staging_seal_draft.json')
S.SEAL_NAME = str(draft)
S.step_seal()  # retains all original checks, including staged test equality
data = json.loads(draft.read_text())
a2 = json.loads((T / 'staging_seal_a2.json').read_text())
data['order_commit'] = ORDER_COMMIT
data['order_sha256'] = S.sha(R / ORDER)
data['base_commit'] = '97edd1fd'
data['a3_note'] = ('A3 order; original stage.py seal checks; route (a): evidence.py ORDER/SEAL_NAME re-pointed to '
                   'this seal; audit routing in a3/sitecustomize.py restricts only the output root')
for name, digest in a2['registration_files'].items():
    if name.startswith('artifacts/eng1/a2/'):
        assert S.sha(R / name) == digest, 'A2 registered file changed: ' + name
        data['registration_files'][name] = digest
for name in A3_REGISTERED:
    data['registration_files'][name] = S.sha(R / name)
# Route (a) changes exactly one sealed tool file; everything else must equal the A2 seal.
assert data['arms'] == a2['arms'], 'arm maps differ from the A2 seal'
assert data['engine_sha256'] == a2['engine_sha256'], 'engine pins differ from the A2 seal'
assert data['module_name'] == a2['module_name'] and data['arms_pending'] == a2['arms_pending'] == []
assert set(data['tool_files']) == set(a2['tool_files']), 'tool file set differs from the A2 seal'
changed = sorted(n for n in data['tool_files'] if data['tool_files'][n] != a2['tool_files'][n])
assert changed == ['evidence.py'], changed
assert {k: v for k, v in data['registration_files'].items() if k not in A3_REGISTERED} == a2['registration_files'], \
    'registration files differ from the A2 seal outside the A3 additions'
dest = T / 'staging_seal_a3.json'
with dest.open('x') as f:
    f.write(json.dumps(data, indent=1, sort_keys=True) + '\n')
print('new seal:', dest.relative_to(R), 'sha256:', S.sha(dest), 'changed tool files vs A2:', changed)
