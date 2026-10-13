"""P1T-FIX7 sweep (i) row i19 receipt: one arm (429446d's before tree, or this worktree) under the LD_PRELOAD
pthread_mutex_lock shim of tensor_cuda/tests/test_dlpack.py (_FIX7_LOCK_SHIM_C, compiled from this worktree's copy).
  A. in-process, no GPU: arm 1 failure, call _C._dlpack_begin_shutdown(5.0); print what it raised / returned and
     whether admission is closed.
  B. ordinary exit (GPU): this worktree's _FIX7_CORE_EXIT_SCRIPT, mode lock_fail 1 (its _report reads the stats with
     .get so the before arm's missing FIX7 keys do not matter); print rc, whether the deleter finished before
     finalization, and the stderr lines.
Usage: lock_failure_receipt.py <arm root> <A|B>. B must run under the GPU lock. No prior art claimed (receipt plumbing;
the shim's prior art is at its definition in test_dlpack.py).
"""
import os
import subprocess
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7')
root, part = Path(sys.argv[1]), sys.argv[2]
src = (W / 'tensor_cuda/tests/test_dlpack.py').read_text()


def literal(name):
    i = src.index(f'{name} = r"""') + len(f'{name} = r"""')
    return src[i:src.index('"""', i)]


shim_dir = Path('/tmp/p1t_fix7_scratch/lockshim_receipt')
shim_dir.mkdir(parents=True, exist_ok=True)
(shim_dir / 'mutex_lock_shim.c').write_text(literal('_FIX7_LOCK_SHIM_C'))
so = shim_dir / 'mutex_lock_shim.so'
subprocess.run(['gcc', '-shared', '-fPIC', '-O1', '-o', str(so), str(shim_dir / 'mutex_lock_shim.c'), '-ldl'],
               check=True)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', LD_PRELOAD=str(so),
           PYTHONPATH=f'{root}/tensor_cuda/tensor_cuda:{root}/tensor_cuda:{root}/tc_signal')
print(f'arm root: {root}; shim: {so}')
if part == 'A':
    code = r'''
import ctypes, tensor_cuda as tc
print("package", tc.__file__)
L = ctypes.CDLL(None)
L.shim_lock_arm(1)
try:
    d = tc._C._dlpack_begin_shutdown(5.0)
    print("RETURNED waited_us=%d" % d["waited_us"])
except Exception as e:
    print("RAISED %s: %s" % (type(e).__name__, e))
st = tc._C._dlpack_release_stats()
print("lock failures injected %d, left %d; admission closed: %s; shutdown_lock_retries: %s"
      % (L.shim_lock_failed(), L.shim_lock_left(), st["closing"], st.get("shutdown_lock_retries", "(no such key)")))
'''
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60,
                       env=dict(env, CUDA_VISIBLE_DEVICES=''))
else:
    i = src.index('_FIX7_CORE_EXIT_SCRIPT = r"""') + len('_FIX7_CORE_EXIT_SCRIPT = r"""')
    sep = '""" + _SUBPROC_PRODUCER + r"""'
    j = src.index(sep, i)
    k = src.index('"""', j + len(sep))
    script = src[i:j] + literal('_SUBPROC_PRODUCER') + src[j + len(sep):k]
    script = script.replace('{k: st[k] for k in', '{k: st.get(k) for k in')
    r = subprocess.run([sys.executable, '-c', script, 'lock_fail', '1'], capture_output=True, text=True, timeout=120,
                       env=env)
    out = r.stdout
    print('deleter finished before finalization:', 0 <= out.find('DELETER finished') < out.find('FINALIZING True'))
print('rc', r.returncode)
print('--- stdout'); print(r.stdout.strip())
print('--- stderr'); print(r.stderr.strip()[-3000:])
