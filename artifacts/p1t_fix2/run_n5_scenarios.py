"""P1T-FIX2: run the N5 exit-coordination scenarios of tensor_cuda/tests/test_dlpack.py (the module's own script text,
unchanged) against a chosen core build, and print each scenario's return code, RESULT line and stderr. Used for the
d116ac0 before-witness (where the test assertions cannot apply: that build has no exit hook and no lifecycle stats).
Usage: PYTHONPATH=<core pkg paths> run_n5_scenarios.py <worktree> <mode>...
No prior art claimed (a runner).
"""
import importlib.util
import os
import subprocess
import sys

w = sys.argv[1]
spec = importlib.util.spec_from_file_location('td', os.path.join(w, 'tensor_cuda/tests/test_dlpack.py'))
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)
import tensor_cuda
print('core module:', tensor_cuda._C.__file__, flush=True)
for mode in sys.argv[2:]:
    r = subprocess.run([sys.executable, '-c', td._FIX2_N5_SCRIPT, mode], capture_output=True, text=True, timeout=120,
                       env=td._subproc_env())
    print(f'== mode={mode} rc={r.returncode}')
    print('stdout:', r.stdout.strip())
    print('stderr:', r.stderr.strip()[-2000:])
