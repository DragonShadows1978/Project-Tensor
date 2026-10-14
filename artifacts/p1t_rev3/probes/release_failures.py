"""LEAD GPU ONLY. PASS iff all FIX1/FIX2 release-failure regressions execute, exit0, and none skip.
Decides OOM at pending push/recycle/error-message, fallback/bound CUDA failures, partial launches,
raising native producer deleters and pending exit release. Prior art: C++17/noexcept; CUDA12.6;
Linux failslab/SQLite allocation failure testing. Taken: existing real-source tests; ours: lead dispatch only.
Run under the lead's GPU lock, timeout600. No source writes, pytest cache disabled.
"""
import os, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTHONPATH']=str(root/'tc_signal')+os.pathsep+str(root/'tensor_cuda')+os.pathsep+env.get('PYTHONPATH','')
r=subprocess.run(['timeout','600',sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
 str(root/'tc_signal/tests/test_fix1_findings.py'),str(root/'tc_signal/tests/test_fix2_findings.py')],
 cwd=root,env=env,capture_output=True,text=True)
print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
assert r.returncode==0 and 'skipped' not in r.stdout and 'passed' in r.stdout,'release gate did not pass without skips'
