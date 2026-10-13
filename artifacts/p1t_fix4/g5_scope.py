"""P1T-FIX4 G5 scope check (read-only git). Checks:
  1. every path changed by this order (working tree vs HEAD 3c7dc22: tracked diff, untracked files, and ignored files
     outside build outputs) is in the order's Exercises paths or docs/P1T_FIX4_*.md, artifacts/p1t_fix4/**, logs/**;
  2. HEAD vs 62c8d9f adds only the lead's round-4 receipt / order commits (listed, not this seat's);
  3. in the diff against 62c8d9f (tensor_cuda/, tc_signal/, comment lines excluded): no new core `__global__`
     (tensor_cuda/; a tc_signal TEST helper kernel is listed separately), no device atomics, fast-math, TF32,
     cudaDeviceSynchronize or host-callback launches; no os._exit / _exit / abort / std::terminate added to PRODUCT code
     (tests excluded); no throw / raise_* call inside any destructor body touched by FIX4 (CheckedDeviceGuard,
     DeviceGuard, QuietDeviceGuard, ~RetainUntilComplete) nor in note_push / note_pop / pending_over_bound;
  4. the reviewer probe files (rounds 1-4, and every committed round-4 receipt) are byte-identical to HEAD, and the
     aggregate sha256 of the probe files equals the one registered in the ledger before any run.
No prior art claimed (a checker; FIX3's g4_scope.py adapted).
"""
import fnmatch
import hashlib
import re
import subprocess
import sys
from pathlib import Path

W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix4'
REGISTERED_PROBE_AGG = '7d6392b4e4fd1b3a011f5f083b5bad19837bd0e3f511c9b4ff75e66a79e8cb92'


def git(*a):
    return subprocess.run(['git', '-C', W, *a], capture_output=True, text=True, check=True).stdout


ALLOWED = ['tensor_cuda/src/bindings.cpp', 'tensor_cuda/src/kernels.cu', 'tensor_cuda/tensor_cuda/__init__.py',
           'tensor_cuda/tests/test_dlpack.py', 'tc_signal/src/*', 'tc_signal/tc_signal/*', 'tc_signal/tests/*',
           'tc_signal/README.md', 'docs/P1T_FIX4_*.md', 'artifacts/p1t_fix4/*', 'logs/*']
BUILD = ['tensor_cuda/build/*', 'tc_signal/build/*', '*.so', '*__pycache__*']
changed = [ln for ln in git('diff', '--name-only', 'HEAD').splitlines() if ln]
untracked = [ln for ln in git('ls-files', '--others', '--exclude-standard').splitlines() if ln]
ignored = [ln for ln in git('ls-files', '--others', '--ignored', '--exclude-standard').splitlines()
           if ln and not any(fnmatch.fnmatch(ln, b) for b in BUILD)]
ok = True
print('1. paths changed by P1T-FIX4 (tracked vs HEAD, untracked, ignored non-build):')
for tag, lst in (('M', changed), ('?', untracked), ('!', ignored)):
    for p in lst:
        inside = any(fnmatch.fnmatch(p, a) for a in ALLOWED)
        ok &= inside
        print(f'   {"OK " if inside else "OUT"} {tag} {p}')
print('2. HEAD vs 62c8d9f (lead commits already on voice-fix4, not this seat\'s):')
for ln in git('log', '--oneline', '62c8d9f..HEAD').splitlines():
    print('   ' + ln[:110])
lead_files = git('diff', '--name-only', '62c8d9f', 'HEAD').split()
lead_outside = [f for f in lead_files if not (f.startswith('artifacts/p1t_rev4/') or f.startswith('logs/p1t_rev4')
                                              or f.startswith('docs/P1T_REV4_') or f.startswith('orders/'))]
print(f'   {len(lead_files)} files, all under artifacts/p1t_rev4/, logs/p1t_rev4*, docs/P1T_REV4_*, orders/: '
      f'{not lead_outside} {lead_outside}')
ok &= not lead_outside
diff = git('diff', '62c8d9f', '--', 'tensor_cuda', 'tc_signal')
# untracked new files count as fully added
for f in untracked:
    if f.startswith(('tensor_cuda/', 'tc_signal/')):
        diff += '\n+++ b/' + f + '\n' + ''.join('+' + l + '\n' for l in Path(W, f).read_text().splitlines())
added, cur = [], None
for ln in diff.splitlines():
    if ln.startswith('+++ '):
        cur = ln[6:]
    elif ln.startswith('+') and not ln.startswith('+++'):
        added.append((cur, ln[1:]))
code = [(f, ln) for f, ln in added if not re.match(r'\s*(//|#|\*)', ln) and '"""' not in ln]
product = [(f, ln) for f, ln in code if '/tests/' not in f]
checks = {
    'new core __global__ (tensor_cuda/)': [ln for f, ln in added if '__global__' in ln and f.startswith('tensor_cuda/')],
    'device atomics (atomicAdd/atomicCAS/...)': [ln for f, ln in code if re.search(r'\batomic(Add|CAS|Exch|Max|Min|Sub|Inc|Dec|And|Or|Xor)\b', ln)],
    'fast-math flags': [ln for f, ln in code if re.search(r'fast.?math|ftz=true|prec-div=false', ln, re.I)],
    'TF32': [ln for f, ln in code if 'TF32' in ln or 'tf32' in ln],
    'cudaDeviceSynchronize': [ln for f, ln in code if 'cudaDeviceSynchronize' in ln],
    'host-callback launches': [ln for f, ln in code if 'cudaLaunchHostFunc' in ln or 'cudaStreamAddCallback' in ln],
    'os._exit / _exit / abort / std::terminate in product code': [
        ln for f, ln in product if re.search(r'os\._exit|\b_exit\s*\(|\babort\s*\(|std::terminate\s*\(', ln)],
}
print('3. added-code checks vs 62c8d9f (tensor_cuda/, tc_signal/; untracked new files included):')
test_kernels = [(f, ln.strip()) for f, ln in added if '__global__' in ln and '/tests/' in f]
print(f'   tc_signal TEST helper kernels (not core, not product): {test_kernels}')
for k, v in checks.items():
    ok &= not v
    print(f'   {k}: {len(v)}' + ('' if not v else ' -> ' + ' | '.join(x.strip()[:80] for x in v[:3])))
SRC = {'dlpack': Path(W, 'tc_signal/src/dlpack_operand.cpp').read_text(), 'guard': Path(W, 'tc_signal/src/cuda_guard.h').read_text(),
       'kernels': Path(W, 'tensor_cuda/src/kernels.cu').read_text()}
src = SRC['dlpack']


def body(sig, text=None):
    src_ = text if text is not None else src
    i = src_.index(sig)
    j = src_.index('{', i)
    depth, k = 0, j
    while True:
        depth += {'{': 1, '}': -1}.get(src_[k], 0)
        if depth == 0:
            return src_[j:k + 1]
        k += 1


def strip_caught(b):
    """Remove every `try { ... } catch (...) { ... }` region: a throw inside it cannot leave the function. (r2: the r1
    checker flagged FIX2's pre-existing test-injection throw, which its own catch (...) catches in the destructor.)"""
    out, exempt = b, []
    while 'try {' in out:
        i = out.index('try {')
        j, depth = out.index('{', i), 0
        k = j
        while True:
            depth += {'{': 1, '}': -1}.get(out[k], 0)
            if depth == 0:
                break
            k += 1
        rest = out[k + 1:]
        m = re.match(r'\s*catch \(\.\.\.\) \{', rest)
        if not m:
            break
        c = k + 1 + m.end() - 1
        depth, e = 0, c
        while True:
            depth += {'{': 1, '}': -1}.get(out[e], 0)
            if depth == 0:
                break
            e += 1
        exempt += [ln.strip() for ln in out[i:k].splitlines() if re.search(r'\bthrow\b', ln)]
        out = out[:i] + out[e + 1:]
    return out, exempt


for key, sig in (('dlpack', 'RetainUntilComplete::~RetainUntilComplete()'), ('dlpack', 'void note_push(PendingState& st'),
                 ('dlpack', 'void note_pop(PendingState& st'), ('dlpack', 'bool pending_over_bound('),
                 ('dlpack', 'void note_unraisable_device_restore_failure('), ('dlpack', 'void destroy_event_on_device('),
                 ('guard', '~DeviceGuard()'), ('guard', '~QuietDeviceGuard()'), ('kernels', '~CheckedDeviceGuard()'),
                 ('kernels', 'bool drain_device_legacy_stream(int dev)'), ('kernels', 'Storage::~Storage()')):
    b = '\n'.join(l for l in body(sig, SRC[key]).splitlines() if not l.strip().startswith('//'))
    b, exempt = strip_caught(b)
    bad = re.findall(r'\bthrow\b|\bck_(?:add|mul|sub)\b|refuse_|overflow_refusal|raise_cuda_failure', b)
    ok &= not bad
    print(f'   no escaping throw / ck_* / refusal in {sig.split("(")[0]}: {not bad} {bad}'
          + (f' (caught locally by catch (...), pre-existing FIX2 code: {exempt})' if exempt else ''))
probes = git('diff', '--stat', 'HEAD', '--', 'artifacts/p1t_rev4', 'artifacts/p1t_rev3', 'artifacts/p1t_rev2', 'artifacts/p1t_rev').strip()
print('4. reviewer probe + receipt files (rounds 1-4) vs HEAD:', 'byte-identical' if not probes else 'CHANGED: ' + probes)
ok &= not probes
# bash, as at registration (r1 used /bin/sh = dash, whose glob order differs from bash's locale collation: same lines,
# different order, different aggregate)
lines = subprocess.run(['bash', '-c', 'sha256sum artifacts/p1t_rev/probes/* artifacts/p1t_rev2/probes/* '
                        'artifacts/p1t_rev3/probes/* artifacts/p1t_rev4/probes/*'], cwd=W, capture_output=True, text=True,
                       check=True).stdout
agg = hashlib.sha256(lines.encode()).hexdigest()
print(f'   aggregate probe sha256 {agg} == registered {REGISTERED_PROBE_AGG}: {agg == REGISTERED_PROBE_AGG}')
ok &= agg == REGISTERED_PROBE_AGG
print('G5', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
