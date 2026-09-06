#!/usr/bin/env python3
"""Run G1 without hiding legacy no-device failures. Receipts are append-only."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from apa_sp1_cpu import registration,sha,write_receipt
from apa_sp1_gpu import load_runtime


def main():
    registration();tc=load_runtime()
    stamp=str(time.time_ns());dest=ROOT/'artifacts/apa_sp1'
    old=['test_apa_selective.py','test_apa_selective_splitk.py','test_apa_selective_int4.py',
         'test_apa_value_dim.py','test_apa_phase6.py','test_qtile_attention.py','test_apa_int4_sdpa_noncausal.py']
    new=['test_apa_sp1_reference.py','test_apa_sp1_host.py','test_apa_sp1_harness.py']
    sources={f:sha(ROOT/'tensor_cuda/tests'/f) for f in old+['test_selector_accuracy.py']}
    env=dict(os.environ,PYTHONPATH=f'{dest}/build:{ROOT}/tensor_cuda',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    results={}
    for name,files in [('cpu',new),('legacy',old)]:
        log=dest/f'g1_{name}_{stamp}.log'
        cmd=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--tb=short']+[str(ROOT/'tensor_cuda/tests'/f) for f in files]
        with log.open('x') as f:
            p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
        results[name]=dict(returncode=p.returncode,log=str(log.relative_to(ROOT)),command=cmd,
                           summary=log.read_text().splitlines()[-1])
        print(name,results[name]['summary'],flush=True)
    # The standalone selector changes only its unsafe hard-coded import path
    # in memory; do not import it during pytest collection (argv is not pytest).
    path=ROOT/'tensor_cuda/tests/test_selector_accuracy.py'
    code="""import sys
from pathlib import Path
root=Path.cwd();path=root/'tensor_cuda/tests/test_selector_accuracy.py'
source=path.read_text();needle='sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")'
assert source.count(needle)==1
source=source.replace(needle,'sys.path.insert(0, '+repr(str(root/'tensor_cuda'))+')')
sys.argv=[str(path),'4']
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
"""
    log=dest/f'g1_selector_{stamp}.log'
    with log.open('x') as f:
        p=subprocess.run([sys.executable,'-c',code],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=30)
    results['selector']=dict(returncode=p.returncode,log=str(log.relative_to(ROOT)),summary=log.read_text().splitlines()[-1],
                             transform='Only hard-coded sys.path changed in memory to this worktree.')
    assert sources=={f:sha(ROOT/'tensor_cuda/tests'/f) for f in sources},'existing tests changed'
    write_receipt(dest/f'g1_{stamp}.json',dict(evidence_class='CPU pytest suite run; legacy CUDA gates attempted and blocked',
         status='CPU_PASS_LEGACY_BLOCKED' if results['cpu']['returncode']==0 and results['legacy']['returncode'] else 'INSPECT_RESULTS',
         runtime_module=str(tc._C.__file__),source_pins=sources,results=results))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
