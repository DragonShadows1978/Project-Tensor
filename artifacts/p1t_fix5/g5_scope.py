"""P1T-FIX5 G5 scope check (read-only git), adapted from FIX4's artifacts/p1t_fix4/g5_scope.py. Checks:
  1. every path changed by this order (working tree vs b2d7918: tracked diff, untracked files, and ignored files outside
     build outputs) is in the order's Exercises paths or docs/P1T_FIX5_*.md, artifacts/p1t_fix5/**, logs/**; HEAD
     (3491eeb) vs b2d7918 adds only orders/ (the lead's order commit);
  2. in the diff against 1475fae (tensor_cuda/, tc_signal/; comment lines excluded): no new core `__global__`, no device
     atomics, fast-math, TF32, device-wide syncs (cudaDeviceSynchronize) or host-callback launches; no os._exit / _exit /
     abort / std::terminate in PRODUCT code; no escaping throw in FIX5's destructors / noexcept bodies (NewHandleOwner x2,
     CapsuleRef::drop, ~CallbackSection, leave / leave_count / admit, wait_without_gil, note_core_unraisable,
     note_cublas_handle_destroy_failure, call_deleter_contained) nor in FIX4's (re-checked);
  3. tc_signal is not linked to _tensor_cuda: CMakeLists.txt unchanged vs 1475fae; the built add-on's DT_NEEDED list
     names no _tensor_cuda;
  4. the reviewer probe files (rounds 1-5) are byte-identical to HEAD, and their aggregate sha256 equals the one
     registered in docs/P1T_FIX5_LEDGER.md before any run.
No prior art claimed (a checker).
"""
import fnmatch
import hashlib
import glob
import re
import subprocess
from pathlib import Path

W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix5'
REGISTERED_PROBE_AGG = 'cd4b50e9cb55f80ceec4b21f5d7afc01287cfa876185289365a48f3769909afa'


def git(*a):
    return subprocess.run(['git', '-C', W, *a], capture_output=True, text=True, check=True).stdout


ALLOWED = ['tensor_cuda/src/kernels.cu', 'tensor_cuda/src/bindings.cpp', 'tensor_cuda/tensor_cuda/__init__.py',
           'tensor_cuda/tests/test_dlpack.py', 'tc_signal/src/*', 'tc_signal/tc_signal/*', 'tc_signal/tests/*',
           'tc_signal/README.md', 'docs/P1T_FIX5_*.md', 'artifacts/p1t_fix5/*', 'logs/*']
BUILD = ['tensor_cuda/build/*', 'tc_signal/build/*', '*.so', '*__pycache__*']
ok = True
changed = [ln for ln in git('diff', '--name-only', 'b2d7918').splitlines() if ln]
untracked = [ln for ln in git('ls-files', '--others', '--exclude-standard').splitlines() if ln]
ignored = [ln for ln in git('ls-files', '--others', '--ignored', '--exclude-standard').splitlines()
           if ln and not any(fnmatch.fnmatch(ln, b) for b in BUILD)]
head_vs_base = [ln for ln in git('diff', '--name-only', 'b2d7918', 'HEAD').splitlines() if ln]
print('1. paths changed vs b2d7918 (tracked, untracked, ignored non-build):')
for tag, lst in (('M', changed), ('?', untracked), ('!', ignored)):
    for p in lst:
        inside = any(fnmatch.fnmatch(p, a) for a in ALLOWED) or (tag == 'M' and p in head_vs_base
                                                                  and p.startswith('orders/'))
        ok &= inside
        print(f'   {"OK " if inside else "OUT"} {tag} {p}')
print(f'   HEAD vs b2d7918 (lead order commit): {head_vs_base}; only orders/: '
      f'{all(p.startswith("orders/") for p in head_vs_base)}')
ok &= all(p.startswith('orders/') for p in head_vs_base)
diff = git('diff', '1475fae', '--', 'tensor_cuda', 'tc_signal')
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
    # code lines only: a comment that mentions __global__ (kernels.cu's V5-04 note) is not a kernel
    'new core __global__ (tensor_cuda/)': [ln for f, ln in code if '__global__' in ln and f.startswith('tensor_cuda/')],
    'device atomics': [ln for f, ln in product if re.search(r'\batomic(Add|CAS|Exch|Max|Min|Sub|Inc|Dec|And|Or|Xor)\b', ln)],
    'fast-math flags': [ln for f, ln in code if re.search(r'fast.?math|ftz=true|prec-div=false', ln, re.I)],
    'TF32': [ln for f, ln in code if 'TF32' in ln or 'tf32' in ln],
    'cudaDeviceSynchronize (product)': [ln for f, ln in product if 'cudaDeviceSynchronize' in ln],
    'host-callback launches': [ln for f, ln in code if 'cudaLaunchHostFunc' in ln or 'cudaStreamAddCallback' in ln],
    'os._exit / _exit / abort / std::terminate in product code': [
        ln for f, ln in product if re.search(r'os\._exit|\b_exit\s*\(|\babort\s*\(|std::terminate\s*\(', ln)],
}
print('2. added-code checks vs 1475fae (tensor_cuda/, tc_signal/; untracked new files included):')
for k, v in checks.items():
    ok &= not v
    print(f'   {k}: {len(v)}' + ('' if not v else ' -> ' + ' | '.join(x.strip()[:80] for x in v[:3])))
SRC = {k: Path(W, p).read_text() for k, p in (('dlpack', 'tc_signal/src/dlpack_operand.cpp'),
                                               ('conv', 'tc_signal/src/conv.cu'), ('lstm', 'tc_signal/src/lstm.cu'),
                                               ('guard', 'tc_signal/src/cuda_guard.h'),
                                               ('kernels', 'tensor_cuda/src/kernels.cu'))}


def body(sig, text):
    i = text.index(sig)
    j = text.index('{', i)
    depth, k = 0, j
    while True:
        depth += {'{': 1, '}': -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k + 1]
        k += 1


def strip_try(b):
    """Drop try { ... } catch (...) / catch (const std::...&) { ... } regions whose throw cannot leave (caught)."""
    out = b
    while 'try {' in out:
        i = out.index('try {')
        j, depth = out.index('{', i), 0
        k = j
        while True:
            depth += {'{': 1, '}': -1}.get(out[k], 0)
            if depth == 0:
                break
            k += 1
        m = re.match(r'\s*catch \([^)]*\) \{', out[k + 1:])
        if not m:
            break
        c = k + 1 + m.end() - 1
        depth, e = 0, c
        while True:
            depth += {'{': 1, '}': -1}.get(out[e], 0)
            if depth == 0:
                break
            e += 1
        handler = out[c:e + 1]
        out = out[:i] + ('' if 'throw' not in handler else handler) + out[e + 1:]
    return out


for key, sig in (('conv', '~NewHandleOwner()'), ('lstm', '~NewHandleOwner()'), ('dlpack', 'void CapsuleRef::drop() noexcept'),
                 ('dlpack', '~CallbackSection()'), ('dlpack', 'void leave_count(Lifecycle& lc'),
                 ('dlpack', 'Admission admit(LifecycleKind k) noexcept'), ('dlpack', 'void leave(LifecycleKind k) noexcept'),
                 ('dlpack', 'void wait_without_gil('), ('dlpack', 'void note_cublas_handle_destroy_failure('),
                 ('dlpack', 'void call_deleter_contained(M* m) noexcept'),
                 ('kernels', 'void note_core_unraisable('), ('dlpack', 'RetainUntilComplete::~RetainUntilComplete()'),
                 ('dlpack', 'void drain_completed() noexcept'), ('guard', '~DeviceGuard()'), ('guard', '~QuietDeviceGuard()'),
                 ('kernels', '~CheckedDeviceGuard()'), ('kernels', 'bool drain_device_legacy_stream(int dev)'),
                 ('kernels', 'Storage::~Storage()')):
    b = '\n'.join(l for l in body(sig, SRC[key]).splitlines() if not l.strip().startswith('//'))
    b = strip_try(b)
    bad = re.findall(r'\bthrow\b|refuse_|raise_cuda_failure', b)
    ok &= not bad
    print(f'   no escaping throw in {key}:{sig.split("(")[0]}: {not bad} {bad}')
cm = git('diff', '--stat', '1475fae', '--', 'tc_signal/CMakeLists.txt').strip()
so = glob.glob(W + '/tc_signal/tc_signal/_tc_signal*.so')[0]
needed = subprocess.run(['readelf', '-d', so], capture_output=True, text=True).stdout
links_core = '_tensor_cuda' in needed
print(f'3. tc_signal/CMakeLists.txt unchanged vs 1475fae: {not cm}; DT_NEEDED names _tensor_cuda: {links_core}')
ok &= (not cm) and not links_core
probes = git('diff', '--stat', 'HEAD', '--', 'artifacts/p1t_rev5', 'artifacts/p1t_rev4', 'artifacts/p1t_rev3',
             'artifacts/p1t_rev2', 'artifacts/p1t_rev').strip()
# the exact registration command (bash glob order, as registered in the ledger at 05:19)
agg = subprocess.run(['bash', '-c', 'sha256sum artifacts/p1t_rev{,2,3,4,5}/probes/* | sha256sum'], cwd=W,
                     capture_output=True, text=True).stdout.split()[0]
files = subprocess.run(['bash', '-c', 'ls artifacts/p1t_rev{,2,3,4,5}/probes/*'], cwd=W, capture_output=True,
                       text=True).stdout.split()
print('4. reviewer probe + receipt files (rounds 1-5) vs HEAD:', 'byte-identical' if not probes else 'CHANGED: ' + probes)
print(f'   probe aggregate sha256 {agg} == registered {REGISTERED_PROBE_AGG}: {agg == REGISTERED_PROBE_AGG} '
      f'({len(files)} files)')
ok &= (not probes) and agg == REGISTERED_PROBE_AGG
print('G5', 'PASS' if ok else 'FAIL')
