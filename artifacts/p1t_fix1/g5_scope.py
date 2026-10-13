"""P1T-FIX1 G5 (scope) without git: the files this seat touched, their line diff stat against the pristine snapshot
taken before the first edit (/tmp/p1t_fix1_base: byte-identical to 00590b7 for every exercised path, ledger 21:56),
the Exercises/allowed-path check, new __global__ kernels in core, and the reviewer's probe byte-identity.
The lead should also run the literal `git diff --stat 00590b7` (it additionally lists the P1T-REV receipt commits).
Prior art: difflib (CPython stdlib). Nothing new."""
import difflib
import hashlib
import os
import re
import sys

W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix1'
BASE = '/tmp/p1t_fix1_base'
INT = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_int'   # 00590b7 checkout
REF = os.path.join(BASE, 'probe_sha256_before.txt')   # written before the first edit
IGNORED = re.compile(r'(^|/)(build|__pycache__|\.pytest_cache)/|\.so$')   # tensor_cuda/.gitignore, tc_signal/.gitignore, .gitignore
ALLOWED = [
    r'^tensor_cuda/src/bindings\.cpp$', r'^tensor_cuda/src/kernels\.cu$', r'^tensor_cuda/include/tc/core\.h$',
    r'^tensor_cuda/tensor_cuda/__init__\.py$', r'^tensor_cuda/tests/test_dlpack\.py$',
    r'^tc_signal/src/[^/]+$', r'^tc_signal/tc_signal/[^/]+$', r'^tc_signal/tests/[^/]+$', r'^tc_signal/README\.md$',
    r'^docs/P1T_FIX1_[^/]*\.md$', r'^artifacts/p1t_fix1/', r'^logs/',
]
ref_t = os.path.getmtime(REF)
touched = []
for root, dirs, files in os.walk(W):
    dirs[:] = [d for d in dirs if d not in ('WorkTree', '.git')]
    for f in files:
        p = os.path.join(root, f)
        rel = os.path.relpath(p, W)
        if os.path.getmtime(p) > ref_t and not IGNORED.search(rel):
            touched.append(rel)
touched.sort()


def lines(p):
    try:
        return open(p, encoding='utf-8', errors='replace').read().splitlines()
    except FileNotFoundError:
        return None


rows, tot_i, tot_d, unchanged, out_of_scope = [], 0, 0, [], []
def baseline_path(rel):
    # The pristine snapshot first; files it did not copy (e.g. artifacts/*.ppm) from the 00590b7 checkout.
    for b in (BASE, INT):
        if os.path.exists(os.path.join(b, rel)):
            return os.path.join(b, rel)
    return None


for rel in touched:
    bp = baseline_path(rel)
    if bp is not None and open(bp, 'rb').read() == open(os.path.join(W, rel), 'rb').read():
        unchanged.append(f'{rel} (== {"snapshot" if bp.startswith(BASE) else "00590b7 checkout"})')
        continue                     # rewritten with byte-identical content: no diff
    new = lines(os.path.join(W, rel))
    old = lines(bp) if bp else None
    if old is None:
        ins, dele = len(new), 0
    else:
        ins = dele = 0
        for ln in difflib.unified_diff(old, new, lineterm='', n=0):
            if ln.startswith('+') and not ln.startswith('+++'):
                ins += 1
            elif ln.startswith('-') and not ln.startswith('---'):
                dele += 1
    ok = any(re.search(a, rel) for a in ALLOWED)
    if not ok:
        out_of_scope.append(rel)
    rows.append((rel, ins, dele, 'new' if old is None else 'mod', ok))
    tot_i += ins
    tot_d += dele

print('== diff stat vs pristine snapshot (00590b7-identical for exercised paths); gitignored build products excluded')
code = [r for r in rows if not r[0].startswith(('artifacts/', 'logs/', 'docs/'))]
for rel, i, d, kind, ok in rows:
    print(f' {rel:<62} | {kind} +{i} -{d}{"" if ok else "   OUT OF SCOPE"}')
ci = sum(r[1] for r in code)
cd = sum(r[2] for r in code)
print(f' {len(rows)} files changed, {tot_i} insertions(+), {tot_d} deletions(-)'
      f'   [code/tests/README only: {len(code)} files, +{ci} -{cd}]')
print('== rewritten by the mandated core suite / runs with byte-identical content (no diff):', len(unchanged))
for rel in unchanged:
    print('   ', rel)
print('== out-of-scope paths:', out_of_scope if out_of_scope else 'none')

# New __global__ kernels in core (tensor_cuda .cu/.cuh/.cpp/.h)
added_global = 0
for rel, *_ in rows:
    if rel.startswith('tensor_cuda/') and re.search(r'\.(cu|cuh|cpp|h)$', rel):
        old = lines(os.path.join(BASE, rel)) or []
        new = lines(os.path.join(W, rel))
        for ln in difflib.unified_diff(old, new, lineterm='', n=0):
            if ln.startswith('+') and not ln.startswith('+++') and '__global__' in ln:
                added_global += 1
                print('   added __global__:', rel, ln)
print('== new __global__ lines in core:', added_global)

# Reviewer probes byte-identical
ok = True
for ln in open(REF):
    h, rel = ln.split()
    cur = hashlib.sha256(open(os.path.join(W, rel), 'rb').read()).hexdigest()
    same = cur == h
    ok &= same
    print(f'== probe {rel}: {"identical" if same else "CHANGED"} ({cur[:16]})')
print('G5', 'PASS' if (not out_of_scope and added_global == 0 and ok) else 'FAIL')
sys.exit(0 if (not out_of_scope and added_global == 0 and ok) else 1)
