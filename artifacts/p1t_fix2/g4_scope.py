"""P1T-FIX2 G4 scope check (read-only git). Checks:
  1. every path changed by this order (working tree vs HEAD 4f62b97, plus untracked files) is in the order's
     Exercises paths or docs/P1T_FIX2_*.md, artifacts/p1t_fix2/**, logs/**;
  2. HEAD vs d116ac0 adds only the lead's round-2 receipt/order commits (listed, not mine);
  3. no new core `__global__`, no added device atomics / fast-math / TF32 / cudaDeviceSynchronize / host-callback
     launches / std::terminate or abort calls in the diff against d116ac0 (comment lines excluded);
  4. the reviewer probe files are byte-identical to HEAD (round 1 and round 2).
No prior art claimed (a checker).
"""
import fnmatch
import re
import subprocess
import sys

W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix2'


def git(*a):
    return subprocess.run(['git', '-C', W, *a], capture_output=True, text=True, check=True).stdout


ALLOWED = ['tensor_cuda/src/bindings.cpp', 'tensor_cuda/src/kernels.cu', 'tensor_cuda/include/tc/core.h',
           'tensor_cuda/tensor_cuda/__init__.py', 'tensor_cuda/tests/test_dlpack.py', 'tc_signal/src/*',
           'tc_signal/tc_signal/*', 'tc_signal/tests/*', 'tc_signal/README.md', 'docs/P1T_FIX2_*.md',
           'artifacts/p1t_fix2/*', 'logs/*']
changed = [ln for ln in git('diff', '--name-only', 'HEAD').splitlines() if ln]
untracked = [ln for ln in git('ls-files', '--others', '--exclude-standard').splitlines() if ln]
ok = True
print('1. paths changed by P1T-FIX2 (tracked vs HEAD + untracked, gitignored build outputs excluded):')
for p in changed + untracked:
    inside = any(fnmatch.fnmatch(p, a) for a in ALLOWED)
    ok &= inside
    print(f'   {"OK " if inside else "OUT"} {p}')
print('2. HEAD vs d116ac0 (lead commits already on voice-fix2, not this seat\'s):')
for ln in git('log', '--oneline', 'd116ac0..HEAD').splitlines():
    print('   ' + ln[:110])
print('   files:', ', '.join(git('diff', '--name-only', 'd116ac0', 'HEAD').split()))
diff = git('diff', 'd116ac0', '--', 'tensor_cuda', 'tc_signal')
added = [ln[1:] for ln in diff.splitlines() if ln.startswith('+') and not ln.startswith('+++')]
code = [ln for ln in added if not re.match(r'\s*(//|#|\*)', ln) and '"""' not in ln]
checks = {
    'new core __global__': [ln for ln in added if '__global__' in ln],
    'device atomics (atomicAdd/atomicCAS/...)': [ln for ln in code if re.search(r'\batomic(Add|CAS|Exch|Max|Min|Sub|Inc|Dec|And|Or|Xor)\b', ln)],
    'fast-math flags': [ln for ln in code if re.search(r'fast.?math|ftz=true|prec-div=false', ln, re.I)],
    'TF32': [ln for ln in code if 'TF32' in ln or 'tf32' in ln],
    'cudaDeviceSynchronize': [ln for ln in code if 'cudaDeviceSynchronize' in ln],
    'host-callback launches': [ln for ln in code if 'cudaLaunchHostFunc' in ln or 'cudaStreamAddCallback' in ln],
    'std::terminate / abort calls': [ln for ln in code if re.search(r'std::terminate\s*\(|\babort\s*\(', ln)],
}
print('3. added-code checks vs d116ac0 (tensor_cuda/, tc_signal/):')
for k, v in checks.items():
    ok &= not v
    print(f'   {k}: {len(v)}' + ('' if not v else ' -> ' + ' | '.join(x.strip()[:80] for x in v[:3])))
host_atomic = [ln for ln in code if 'std::atomic' in ln]
print(f'   (informational) host-side std::atomic counter lines added: {len(host_atomic)}')
probes = git('diff', '--stat', 'HEAD', '--', 'artifacts/p1t_rev2', 'artifacts/p1t_rev').strip()
print('4. reviewer probe files vs HEAD:', 'byte-identical' if not probes else 'CHANGED: ' + probes)
ok &= not probes
print('G4', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
