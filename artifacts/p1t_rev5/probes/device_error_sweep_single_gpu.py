"""PASS iff the actual FIX4 core/add-on interposer tests all pass with no skips.
Prior art: CUDA12.6 device/event/error contracts; glibc LD_PRELOAD; reused FIX4 tests.
LEAD ONLY, one GPU, timeout300. Decides export/fill/op device switches, label proof, restores, sticky/prior errors.
"""
import subprocess,os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
env['PYTHONPATH']=str(root/'tensor_cuda')+os.pathsep+str(root/'tc_signal')+os.pathsep+env.get('PYTHONPATH','')
r=subprocess.run(['timeout','290',sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--basetemp=/tmp/p1t_rev5_fix4_pytest',str(root/'tensor_cuda/tests/test_dlpack.py'),str(root/'tc_signal/tests/test_fix4_findings.py'),'-k','fix4'],env=env,text=True,capture_output=True)
print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
assert r.returncode==0 and 'passed' in r.stdout and 'skipped' not in r.stdout,'FIX4 interposer checks did not all pass'
