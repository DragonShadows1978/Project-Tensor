"""P1T-FIX8 G5 scope check (read-only git): a copy of P1T-FIX7's artifacts/p1t_fix7/g5_scope.py retargeted (W = wt_fix8,
BASE = 1cd8824, CODE_BASE = 6991939, FIX8 paths, FIX8's new noexcept / reporting bodies added, round 8 included; the
reviewer probe list registered before any FIX8 change: artifacts/p1t_fix8/probe_files_before.sha256, 114 files). The
FIX6 text follows.
Checks:
  1. every path changed by this order (working tree vs 123610f: tracked diff, untracked files, and ignored files outside
     build outputs) is in the order's Exercises paths or docs/P1T_FIX6_*.md, artifacts/p1t_fix6/**, logs/**; HEAD
     (6ab46a0) vs 123610f adds only orders/ (the lead's order commit);
  2. in the diff against acd42fa (tensor_cuda/, tc_signal/; comment lines excluded): no new core `__global__`, no device
     atomics, fast-math, TF32, device-wide syncs (cudaDeviceSynchronize) or host-callback launches; no os._exit / _exit /
     abort / std::terminate in PRODUCT code; no escaping throw in the destructors / noexcept bodies FIX6 added or changed
     (CapsuleRef::drop, SyncReport::add / merge, clear_reported_records, note_core_unraisable, dlpack_ack_unraisable,
     wait_without_gil) nor in FIX5's list (re-checked);
  3. tc_signal is not linked to _tensor_cuda: CMakeLists.txt unchanged vs acd42fa; the built add-on's DT_NEEDED list
     names no _tensor_cuda;
  4. the reviewer probe files (rounds 1-6, artifacts/p1t_rev*/probes/**) are byte-identical: their per-file sha256 list
     equals the one registered in docs/P1T_FIX6_LEDGER.md before any change (artifacts/p1t_fix6/probe_files_before.sha256,
     82 files), and nothing under artifacts/p1t_rev*/ differs from HEAD.
P1T-FIX7 lead send-back L1: the three new noexcept bodies (wait_device_work_diagnosed, exit_device_waits_diagnosed,
exit_device_waits_undiagnosed) added to the escaping-throw list. No prior art claimed (a checker).
"""
import fnmatch
import glob
import re
import subprocess
from pathlib import Path

W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8'
BASE, CODE_BASE = '1cd8824', '6991939'


def git(*a):
    return subprocess.run(['git', '-C', W, *a], capture_output=True, text=True, check=True).stdout


ALLOWED = ['tensor_cuda/src/kernels.cu', 'tensor_cuda/src/bindings.cpp', 'tensor_cuda/tensor_cuda/__init__.py',
           'tensor_cuda/tests/test_dlpack.py', 'tc_signal/src/*', 'tc_signal/tc_signal/*', 'tc_signal/tests/*',
           'tc_signal/README.md', 'docs/P1T_FIX8_*.md', 'artifacts/p1t_fix8/*', 'logs/*']
BUILD = ['tensor_cuda/build/*', 'tc_signal/build/*', '*.so', '*__pycache__*']
ok = True
changed = [ln for ln in git('diff', '--name-only', BASE).splitlines() if ln]
untracked = [ln for ln in git('ls-files', '--others', '--exclude-standard').splitlines() if ln]
ignored = [ln for ln in git('ls-files', '--others', '--ignored', '--exclude-standard').splitlines()
           if ln and not any(fnmatch.fnmatch(ln, b) for b in BUILD)]
head_vs_base = [ln for ln in git('diff', '--name-only', BASE, 'HEAD').splitlines() if ln]
print(f'1. paths changed vs {BASE} (tracked, untracked, ignored non-build):')
for tag, lst in (('M', changed), ('?', untracked), ('!', ignored)):
    for p in lst:
        inside = any(fnmatch.fnmatch(p, a) for a in ALLOWED) or (tag == 'M' and p in head_vs_base
                                                                  and p.startswith('orders/'))
        ok &= inside
        print(f'   {"OK " if inside else "OUT"} {tag} {p}')
only_orders = all(p.startswith('orders/') for p in head_vs_base)
print(f'   HEAD vs {BASE} (lead order commit): {head_vs_base}; only orders/: {only_orders}')
ok &= only_orders
diff = git('diff', CODE_BASE, '--', 'tensor_cuda', 'tc_signal')
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
    'new core __global__ (tensor_cuda/)': [ln for f, ln in code if '__global__' in ln and f.startswith('tensor_cuda/')],
    'device atomics': [ln for f, ln in product if re.search(r'\batomic(Add|CAS|Exch|Max|Min|Sub|Inc|Dec|And|Or|Xor)\b', ln)],
    # compiler flags only (FIX5's broader `fast.?math` pattern also matches the pre-existing "Fast-math canary"
    # docstring of fp32_divide_probe, which counts as an added line here because FIX6 rewrote that registration)
    'fast-math flags': [ln for f, ln in code if re.search(r'use_fast_math|ffast-math|ftz=true|prec-div=false', ln, re.I)],
    'TF32': [ln for f, ln in code if 'TF32' in ln or 'tf32' in ln],
    'cudaDeviceSynchronize (product)': [ln for f, ln in product if 'cudaDeviceSynchronize' in ln],
    'host-callback launches': [ln for f, ln in code if 'cudaLaunchHostFunc' in ln or 'cudaStreamAddCallback' in ln],
    'os._exit / _exit / abort / std::terminate in product code': [
        ln for f, ln in product if re.search(r'os\._exit|\b_exit\s*\(|\babort\s*\(|std::terminate\s*\(', ln)],
}
print(f'2. added-code checks vs {CODE_BASE} (tensor_cuda/, tc_signal/; untracked new files included; '
      f'{len(code)} added code lines):')
for k, v in checks.items():
    ok &= not v
    print(f'   {k}: {len(v)}' + ('' if not v else ' -> ' + ' | '.join(x.strip()[:80] for x in v[:3])))
SRC = {k: Path(W, p).read_text() for k, p in (('dlpack', 'tc_signal/src/dlpack_operand.cpp'),
                                               ('conv', 'tc_signal/src/conv.cu'), ('lstm', 'tc_signal/src/lstm.cu'),
                                               ('guard', 'tc_signal/src/cuda_guard.h'),
                                               ('kernels', 'tensor_cuda/src/kernels.cu'),
                                               ('cbind', 'tensor_cuda/src/bindings.cpp'))}


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
    """Drop try { ... } catch (...) regions whose throw cannot leave (caught and not rethrown)."""
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


for key, sig in (('dlpack', 'void CapsuleRef::drop() noexcept'), ('dlpack', 'void add(const FailureRecord& r) noexcept'),
                 # P1T-FIX7: the new / changed noexcept recording and exit bodies
                 ('dlpack', 'void sat_inc(unsigned long long& c) noexcept'), ('dlpack', 'void sat_add('),
                 ('dlpack', 'void note_release_failure('), ('dlpack', 'void note_unraisable_failure('),
                 ('dlpack', 'void destroy_event_on_device('), ('dlpack', 'SyncResult sync_device_legacy(int device) noexcept'),
                 ('dlpack', 'void exit_log_unreported() noexcept'), ('dlpack', 'void note_unraisable_device_restore_failure('),
                 ('dlpack', 'FailureRecord first() const noexcept'), ('dlpack', 'unsigned long long exit_hook_closes() noexcept'),
                 ('dlpack', 'void merge(const SyncReport& o) noexcept'),
                 ('dlpack', 'void clear_reported_records(PendingState& st) noexcept'),
                 ('kernels', 'void note_core_unraisable('), ('kernels', 'void dlpack_ack_unraisable('),
                 ('dlpack', 'void wait_without_gil('),
                 # P1T-FIX8: the marker record, the watchdog, the exit report and both allocation-free fallbacks
                 ('dlpack', 'void note_exit_marker_failure('), ('dlpack', 'void destroy_marker('),
                 ('dlpack', 'bool wait_device_work_diagnosed('), ('dlpack', 'void device_wait_line('),
                 ('dlpack', 'void device_wait_finished('), ('dlpack', 'void stop() noexcept'),
                 ('dlpack', 'void run() noexcept'), ('dlpack', 'void diagnose() noexcept'),
                 ('dlpack', 'ReportSummary summarize_unreported('),
                 ('dlpack', 'void write_summary_line('), ('dlpack', 'void note_raised_at_exit('),
                 ('dlpack', 'void exit_report_written() noexcept'), ('dlpack', 'void exit_log_unwritten_report() noexcept'),
                 ('dlpack', 'void exit_runner_window('), ('kernels', 'void dlpack_unreported_unraisable_raw('),
                 ('cbind', 'void write_core_summary('), ('cbind', 'void core_exit_last_reader('),
                 # P1T-FIX7 lead send-back L1: the diagnosed device wait and its window (noexcept)
                 ('dlpack', 'void exit_device_waits_diagnosed('),
                 ('dlpack', 'void exit_device_waits_undiagnosed('),
                 # FIX5's list, re-checked
                 ('conv', '~NewHandleOwner()'), ('lstm', '~NewHandleOwner()'), ('dlpack', '~CallbackSection()'),
                 ('dlpack', 'void leave_count(Lifecycle& lc'), ('dlpack', 'Admission admit(LifecycleKind k) noexcept'),
                 ('dlpack', 'void leave(LifecycleKind k) noexcept'), ('dlpack', 'void note_cublas_handle_destroy_failure('),
                 ('dlpack', 'void call_deleter_contained(M* m) noexcept'),
                 ('dlpack', 'RetainUntilComplete::~RetainUntilComplete()'), ('dlpack', 'void drain_completed() noexcept'),
                 ('guard', '~DeviceGuard()'), ('guard', '~QuietDeviceGuard()'), ('kernels', '~CheckedDeviceGuard()'),
                 ('kernels', 'bool drain_device_legacy_stream(int dev)'), ('kernels', 'Storage::~Storage()')):
    b = '\n'.join(l for l in body(sig, SRC[key]).splitlines() if not l.strip().startswith('//'))
    b = strip_try(b)
    bad = re.findall(r'\bthrow\b|refuse_|raise_cuda_failure', b)
    ok &= not bad
    print(f'   no escaping throw in {key}:{sig.split("(")[0]}: {not bad} {bad}')
cm = git('diff', '--stat', CODE_BASE, '--', 'tc_signal/CMakeLists.txt').strip()
so = glob.glob(W + '/tc_signal/tc_signal/_tc_signal*.so')[0]
needed = subprocess.run(['readelf', '-d', so], capture_output=True, text=True).stdout
links_core = '_tensor_cuda' in needed
print(f'3. tc_signal/CMakeLists.txt unchanged vs {CODE_BASE}: {not cm}; DT_NEEDED names _tensor_cuda: {links_core}')
ok &= (not cm) and not links_core
rev_dirs = ['artifacts/p1t_rev', 'artifacts/p1t_rev2', 'artifacts/p1t_rev3', 'artifacts/p1t_rev4', 'artifacts/p1t_rev5',
            'artifacts/p1t_rev6', 'artifacts/p1t_rev7', 'artifacts/p1t_rev8']
rev_changed = git('diff', '--stat', 'HEAD', '--', *rev_dirs).strip()
rev_untracked = [ln for ln in git('ls-files', '--others', '--exclude-standard', '--', *rev_dirs).splitlines() if ln]
registered = Path(W, 'artifacts/p1t_fix8/probe_files_before.sha256').read_text().splitlines()
files = [ln.split(None, 1)[1] for ln in registered]
now = subprocess.run(['sha256sum', *files], cwd=W, capture_output=True, text=True).stdout.splitlines()
same = now == registered
current_list = sorted(str(p.relative_to(W)) for d in rev_dirs for p in Path(W, d).glob('probes/**/*') if p.is_file())
print(f'4. reviewer probe files: per-file sha256 equal to the registered list: {same} ({len(files)} files; probe '
      f'files present now: {len(current_list)}, same set: {current_list == sorted(files)}); artifacts/p1t_rev*/ vs HEAD: '
      f'{"unchanged" if not rev_changed and not rev_untracked else "CHANGED " + rev_changed + str(rev_untracked)}')
ok &= same and current_list == sorted(files) and not rev_changed and not rev_untracked
print('G5', 'PASS' if ok else 'FAIL')
