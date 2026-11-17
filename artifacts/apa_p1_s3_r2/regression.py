"""Exact pytest phase outcomes, with external read census.
Prior art: pytest hook API, inherited P1 per-ID comparison (2026).
"""
from pathlib import Path
import os,sys,json,hashlib,subprocess
A=Path('artifacts/apa_p1_s3_r2').resolve();ROOT=Path.cwd()
class Receipt:
    def __init__(self,arm):self.arm=arm;self.rows={};self.collection=[];self.errors=[];self.reads=set()
    def audit(self,event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            try:p=Path(os.fsdecode(args[0])).resolve()
            except (ValueError,OSError):return
            if not p.is_relative_to(ROOT):self.reads.add(str(p).replace(str(Path.home()),'$HOME'))
    def pytest_sessionstart(self,session):
        import tensor_cuda as tc
        p=Path(tc._C.__file__).resolve();self.module=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),python=str(Path(tc.__file__).resolve()))
    def pytest_collection_finish(self,session):self.collection=[i.nodeid for i in session.items]
    def pytest_collectreport(self,report):
        if report.failed:self.errors.append(dict(id=report.nodeid,message=str(report.longrepr)))
    def pytest_runtest_logreport(self,report):
        self.rows.setdefault(report.nodeid,[]).append(dict(when=report.when,outcome=report.outcome,message=str(report.longrepr) if report.longrepr else ''))
    def pytest_sessionfinish(self,session,exitstatus):
        r=dict(arm=self.arm,module=self.module,exit=int(exitstatus),collection=self.collection,collection_errors=self.errors,reports=self.rows,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),external_reads=sorted(self.reads))
        (A/f'OUTCOMES_{self.arm}.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':
    arm=sys.argv[1];assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if len(sys.argv)>2 and sys.argv[2]=='--child':
        import pytest,numpy as np
        np.random.seed(142);plugin=Receipt(arm);sys.addaudithook(plugin.audit)
        files=json.loads((A/'REGRESSION_FILES.json').read_text())
        raise SystemExit(pytest.main(['-q','--tb=short','-ra','-p','no:cacheprovider','--continue-on-collection-errors','--basetemp',f'/tmp/apap1s3r2_regression_{arm}/pytest',*files],plugins=[plugin]))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONPATH=str(A/'refs'/arm)+':'+str(ROOT/'tensor_cuda/tests')+':'+str(ROOT/'artifacts/apa_p1'),TMPDIR=f'/tmp/apap1s3r2_regression_{arm}',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    with (A/f'regression_{arm}.log').open('x') as f:
        p=subprocess.run(['timeout','530s',sys.executable,'-B',__file__,arm,'--child'],env=env,stdout=f,stderr=subprocess.STDOUT)
    print('pytest exit',p.returncode,'receipt exists',(A/f'OUTCOMES_{arm}.json').exists())
    raise SystemExit(0 if (A/f'OUTCOMES_{arm}.json').exists() else 1)
