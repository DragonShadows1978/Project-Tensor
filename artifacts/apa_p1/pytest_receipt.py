# Prior art: pytest hook/report API (pytest project); taken for per-ID census.
import json,hashlib,os
from pathlib import Path
class Receipt:
    def __init__(self,path): self.path=Path(path); self.rows={}; self.collection=[]; self.errors=[]
    def pytest_sessionstart(self,session):
        import tensor_cuda as tc
        p=Path(tc._C.__file__).resolve()
        self.module=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),python=str(Path(tc.__file__).resolve()),packed=hasattr(tc,'apa_eb3'))
    def pytest_collection_finish(self,session): self.collection=[i.nodeid for i in session.items]
    def pytest_collectreport(self,report):
        if report.failed:self.errors.append(dict(nodeid=report.nodeid,message=str(report.longrepr)))
    def pytest_runtest_logreport(self,report):
        self.rows.setdefault(report.nodeid,[]).append(dict(when=report.when,outcome=report.outcome,message=str(report.longrepr) if report.longrepr else '',duration=report.duration))
    def pytest_sessionfinish(self,session,exitstatus):
        self.path.write_text(json.dumps(dict(module=self.module,exit=int(exitstatus),collection=self.collection,collection_errors=self.errors,reports=self.rows,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),indent=2)+'\n')
if __name__=='__main__':
    import sys,pytest,numpy as np
    np.random.seed(142)
    p=Receipt(sys.argv[1]); raise SystemExit(pytest.main(sys.argv[2:],plugins=[p]))
