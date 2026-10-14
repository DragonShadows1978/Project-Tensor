"""P1T-FIX7 lead send-back L1 receipt: the three exit waits L1 made diagnosed, staged exactly as this worktree's tests
stage them, on one arm's packages (an arm root holding tensor_cuda/ and tc_signal/ with their built .so files):
  core_lock : test_dlpack.py _FIX7_CORE_EXIT_SCRIPT, mode lock_fail "15:0.05" (15 failed lifecycle-mutex locks at the
              core's exit close, first-diagnostic delay 0.05 s; a deleter in flight); its _report reads the stats with
              .get, so an arm without the L1 keys still reports.
  addon_poll: test_fix7_findings.py _EXIT_SCRIPT, mode lock_fail "1" (the add-on exit wait's lifecycle mutex fails
              once: polling fallback; delay 0.05 s; a deleter in flight).
  addon_dev : test_fix7_findings.py _DEVICE_WAIT_SCRIPT, mode diag (~0.6 s of GPU work queued at exit, delay 0.05 s;
              stats read with .get, as above).
For each: rc, whether the deleter finished before finalization (where staged), and the count of each diagnostic line.
Usage: l1_receipt.py <arm root> <core_lock|addon_poll|addon_dev>. GPU stagings: run under the GPU lock. The shims are
compiled from this worktree's test sources. No prior art claimed (receipt plumbing; the stagings' prior art is at
their definitions).
"""
import os
import subprocess
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7')
root, which = Path(sys.argv[1]), sys.argv[2]
core_src = (W / 'tensor_cuda/tests/test_dlpack.py').read_text()
addon_src = (W / 'tc_signal/tests/test_fix7_findings.py').read_text()
scratch = Path('/tmp/p1t_fix7_scratch/l1_receipt')
scratch.mkdir(parents=True, exist_ok=True)


def lit(src, name, q):
    i = src.index(f'{name} = r{q}') + len(f'{name} = r{q}')
    return src[i:src.index(q, i)]


def shim(src, name, q, tag):
    c, so = scratch / f'{tag}.c', scratch / f'{tag}.so'
    c.write_text(lit(src, name, q))
    subprocess.run(['gcc', '-shared', '-fPIC', '-O1', '-o', str(so), str(c), '-ldl'], check=True)
    return str(so)


env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH=f'{root}/tensor_cuda/tensor_cuda:{root}/tensor_cuda:{root}/tc_signal')
print(f'arm root: {root}; staging: {which}')
for so in sorted(root.glob('tensor_cuda/tensor_cuda/_tensor_cuda*.so')) + sorted(root.glob('tc_signal/tc_signal/_tc_signal*.so')):
    print('binary', subprocess.run(['sha256sum', str(so)], capture_output=True, text=True).stdout.strip())
if which == 'core_lock':
    q = '"""'
    i = core_src.index('_FIX7_CORE_EXIT_SCRIPT = r"""') + len('_FIX7_CORE_EXIT_SCRIPT = r"""')
    sep = '""" + _SUBPROC_PRODUCER + r"""'
    j = core_src.index(sep, i)
    k = core_src.index(q, j + len(sep))
    script = core_src[i:j] + lit(core_src, '_SUBPROC_PRODUCER', q) + core_src[j + len(sep):k]
    script = script.replace('{k: st[k] for k in', '{k: st.get(k) for k in')
    so = shim(core_src, '_FIX7_LOCK_SHIM_C', q, 'core_lock_shim')
    r = subprocess.run([sys.executable, '-c', script, 'lock_fail', '15:0.05'], capture_output=True, text=True,
                       timeout=120, env=dict(env, LD_PRELOAD=so))
    marks = {'lock-retry diagnostic lines': 'cannot take the DLPack release lifecycle lock',
             'lock taken after diagnostics': 'took the DLPack release lifecycle lock'}
elif which == 'addon_poll':
    q = "'''"
    script = lit(addon_src, '_EXIT_SCRIPT', q).replace('TESTS_DIR', repr(str(W / 'tc_signal/tests')))
    so = shim(addon_src, '_LOCK_SHIM_C', q, 'addon_lock_shim')
    r = subprocess.run([sys.executable, '-c', script, 'lock_fail', '1'], capture_output=True, text=True, timeout=120,
                       env=dict(env, LD_PRELOAD=so))
    marks = {'polling-fallback diagnostic lines': 'because the lifecycle mutex or condition variable failed',
             'condition-variable diagnostic lines': 'interpreter exit waits for them (no new one is admitted)\n'}
else:
    q = "'''"
    script = lit(addon_src, '_DEVICE_WAIT_SCRIPT', q).replace("{k: st[k] for k in", "{k: st.get(k) for k in").replace(
        "_C.pending_stats()['exit_device_wait_diagnostics']", "_C.pending_stats().get('exit_device_wait_diagnostics')")
    r = subprocess.run([sys.executable, '-c', script, 'diag'], capture_output=True, text=True, timeout=120, env=env)
    marks = {'device-wait diagnostic lines': 'interpreter exit is waiting for the work queued on the legacy stream',
             'device-wait finished lines': 'the work queued on the legacy stream of CUDA device 0 finished after'}
out, err = r.stdout, r.stderr
print('rc', r.returncode)
if 'DELETER finished' in out or 'FINALIZING' in out:
    print('deleter finished before finalization:', 0 <= out.find('DELETER finished') < out.find('FINALIZING True'))
for label, m in marks.items():
    n = sum(1 for ln in (err + '\n').splitlines(True) if m.strip('\n') in ln and (not m.endswith('\n') or ln.rstrip('\n').endswith(m.strip('\n'))))
    print(f'{label}: {n}')
print('--- stdout'); print(out.strip()[-1500:])
print('--- stderr'); print(err.strip()[-2500:])
