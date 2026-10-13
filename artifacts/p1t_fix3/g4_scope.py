"""P1T-FIX3 G4 scope check (read-only git). Checks:
  1. every path changed by this order (working tree vs HEAD 021abbb: tracked diff, untracked files, and ignored files
     outside build outputs) is in the order's Exercises paths or docs/P1T_FIX3_*.md, artifacts/p1t_fix3/**, logs/**;
  2. HEAD vs 5f9f955 adds only the lead's round-3 receipt / order commits (listed, not this seat's);
  3. in the diff against 5f9f955 (tensor_cuda/, tc_signal/, comment lines excluded): no new `__global__`, no device
     atomics, fast-math, TF32, cudaDeviceSynchronize or host-callback launches; no os._exit / _exit / abort /
     std::terminate added to PRODUCT code (tests excluded); and no throw / ck_* / refuse_* call inside the
     ~RetainUntilComplete destructor body, note_push or note_pop (V3-02: nothing throws on the release path);
  4. the reviewer probe files (rounds 1-3, and every committed round-3 receipt) are byte-identical to HEAD, and the
     aggregate sha256 of the probe files equals the one registered in the ledger before any run.
No prior art claimed (a checker; same structure as FIX2's g4_scope.py).
"""
import fnmatch
import hashlib
import re
import subprocess
import sys
from pathlib import Path

W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix3'
REGISTERED_PROBE_AGG = '36502118ca786939c77714f46c4041760d4c8d4b5f73f69e6056347aec97a82b'


def git(*a):
    return subprocess.run(['git', '-C', W, *a], capture_output=True, text=True, check=True).stdout


ALLOWED = ['tensor_cuda/src/bindings.cpp', 'tensor_cuda/tensor_cuda/__init__.py', 'tensor_cuda/tests/test_dlpack.py',
           'tc_signal/src/dlpack_operand.h', 'tc_signal/src/dlpack_operand.cpp', 'tc_signal/src/*_bindings.cpp',
           'tc_signal/src/bindings.cpp', 'tc_signal/tests/*', 'tc_signal/README.md', 'docs/P1T_FIX3_*.md',
           'artifacts/p1t_fix3/*', 'logs/*']
BUILD = ['tensor_cuda/build/*', 'tc_signal/build/*', '*.so', '*__pycache__*']
changed = [ln for ln in git('diff', '--name-only', 'HEAD').splitlines() if ln]
untracked = [ln for ln in git('ls-files', '--others', '--exclude-standard').splitlines() if ln]
ignored = [ln for ln in git('ls-files', '--others', '--ignored', '--exclude-standard').splitlines()
           if ln and not any(fnmatch.fnmatch(ln, b) for b in BUILD)]
ok = True
print('1. paths changed by P1T-FIX3 (tracked vs HEAD, untracked, ignored non-build):')
for tag, lst in (('M', changed), ('?', untracked), ('!', ignored)):
    for p in lst:
        inside = any(fnmatch.fnmatch(p, a) for a in ALLOWED)
        ok &= inside
        print(f'   {"OK " if inside else "OUT"} {tag} {p}')
print('2. HEAD vs 5f9f955 (lead commits already on voice-fix3, not this seat\'s):')
for ln in git('log', '--oneline', '5f9f955..HEAD').splitlines():
    print('   ' + ln[:110])
lead_files = git('diff', '--name-only', '5f9f955', 'HEAD').split()
lead_outside = [f for f in lead_files if not (f.startswith('artifacts/p1t_rev3/') or f.startswith('logs/p1t_rev3')
                                              or f.startswith('docs/P1T_REV3_') or f.startswith('orders/'))]
print(f'   {len(lead_files)} files, all under artifacts/p1t_rev3/, logs/p1t_rev3*, docs/P1T_REV3_*, orders/: '
      f'{not lead_outside} {lead_outside}')
ok &= not lead_outside
diff = git('diff', '5f9f955', '--', 'tensor_cuda', 'tc_signal')
added, cur = [], None
for ln in diff.splitlines():
    if ln.startswith('+++ '):
        cur = ln[6:]
    elif ln.startswith('+') and not ln.startswith('+++'):
        added.append((cur, ln[1:]))
code = [(f, ln) for f, ln in added if not re.match(r'\s*(//|#|\*)', ln) and '"""' not in ln]
product = [(f, ln) for f, ln in code if '/tests/' not in f]
checks = {
    'new core __global__': [ln for f, ln in added if '__global__' in ln],
    'device atomics (atomicAdd/atomicCAS/...)': [ln for f, ln in code if re.search(r'\batomic(Add|CAS|Exch|Max|Min|Sub|Inc|Dec|And|Or|Xor)\b', ln)],
    'fast-math flags': [ln for f, ln in code if re.search(r'fast.?math|ftz=true|prec-div=false', ln, re.I)],
    'TF32': [ln for f, ln in code if 'TF32' in ln or 'tf32' in ln],
    'cudaDeviceSynchronize': [ln for f, ln in code if 'cudaDeviceSynchronize' in ln],
    'host-callback launches': [ln for f, ln in code if 'cudaLaunchHostFunc' in ln or 'cudaStreamAddCallback' in ln],
    'os._exit / _exit / abort / std::terminate in product code': [
        ln for f, ln in product if re.search(r'os\._exit|\b_exit\s*\(|\babort\s*\(|std::terminate\s*\(', ln)],
}
print('3. added-code checks vs 5f9f955 (tensor_cuda/, tc_signal/):')
for k, v in checks.items():
    ok &= not v
    print(f'   {k}: {len(v)}' + ('' if not v else ' -> ' + ' | '.join(x.strip()[:80] for x in v[:3])))
src = Path(W, 'tc_signal/src/dlpack_operand.cpp').read_text()


def body(sig):
    i = src.index(sig)
    j = src.index('{', i)
    depth, k = 0, j
    while True:
        depth += {'{': 1, '}': -1}.get(src[k], 0)
        if depth == 0:
            return src[j:k + 1]
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


for sig in ('RetainUntilComplete::~RetainUntilComplete()', 'void note_push(PendingState& st',
            'void note_pop(PendingState& st', 'bool pending_over_bound('):
    b = '\n'.join(l for l in body(sig).splitlines() if not l.strip().startswith('//'))
    b, exempt = strip_caught(b)
    bad = re.findall(r'\bthrow\b|\bck_(?:add|mul|sub)\b|refuse_|overflow_refusal', b)
    ok &= not bad
    print(f'   no escaping throw / ck_* / refusal in {sig.split("(")[0]}: {not bad} {bad}'
          + (f' (caught locally by catch (...), pre-existing FIX2 code: {exempt})' if exempt else ''))
probes = git('diff', '--stat', 'HEAD', '--', 'artifacts/p1t_rev3', 'artifacts/p1t_rev2', 'artifacts/p1t_rev').strip()
print('4. reviewer probe + receipt files vs HEAD:', 'byte-identical' if not probes else 'CHANGED: ' + probes)
ok &= not probes
# bash, as at registration (r1 used /bin/sh = dash, whose glob order differs from bash's locale collation: same lines,
# different order, different aggregate)
lines = subprocess.run(['bash', '-c', 'sha256sum artifacts/p1t_rev3/probes/* artifacts/p1t_rev2/probes/* '
                        'artifacts/p1t_rev/probes/*'], cwd=W, capture_output=True, text=True, check=True).stdout
agg = hashlib.sha256(lines.encode()).hexdigest()
print(f'   aggregate probe sha256 {agg} == registered {REGISTERED_PROBE_AGG}: {agg == REGISTERED_PROBE_AGG}')
ok &= agg == REGISTERED_PROBE_AGG
print('G4', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
