"""LEAD ONLY. PASS iff selected real reviewed-build tests pass with zero skips; exit3 for unavailable GPUs/skips.
Lane fix56: round5/6 regression and argument/exit/error gates. Lane semantics: finite operators/streams.
Lane multidevice: needs2 GPUs, exits3 on the lead's one GPU. No copied implementation; actual tests build against sources.
Prior art: existing project regression/oracle tests; CPython/DLPack/CUDA/ONNX contracts. No new algorithm.
"""
from pathlib import Path
import sys,os,subprocess,xml.etree.ElementTree as E
import torch
lane=sys.argv[1] if len(sys.argv)>1 else 'fix56'
if not torch.cuda.is_available() or (lane=='multidevice' and torch.cuda.device_count()<2):raise SystemExit(3)
root=Path(__file__).resolve().parents[3];out=root/'artifacts/p1t_rev7';args=[]
if lane=='fix56':args=['tensor_cuda/tests/test_dlpack.py','tc_signal/tests/test_fix5_findings.py','tc_signal/tests/test_fix6_findings.py','-k','fix5 or fix6 or v50']
elif lane=='semantics':args=['tc_signal/tests/test_conv.py','tc_signal/tests/test_lstm.py','tc_signal/tests/test_norms.py','tc_signal/tests/test_small_ops.py','tc_signal/tests/test_contract.py']
elif lane=='multidevice':args=['tensor_cuda/tests/test_dlpack.py','tc_signal/tests/test_fix4_findings.py','-k','device']
else:raise ValueError(lane)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'tensor_cuda')+':'+str(root/'tensor_cuda/tensor_cuda')+':'+str(root/'tc_signal'))
receipt=out/('lead_'+lane+'.xml')
r=subprocess.run(['timeout','900',sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*args,'--junitxml='+str(receipt)],cwd=root,env=env,timeout=910)
if r.returncode:raise SystemExit(r.returncode)
cases=list(E.parse(receipt).iter('testcase'))
if not cases or any(list(x.iter('skipped')) for x in cases):print('INCONCLUSIVE: absent tests or skips');raise SystemExit(3)
assert all(not list(x.iter('failure')) and not list(x.iter('error')) for x in cases)
print('PASS',lane,len(cases),'tests with zero skips')
