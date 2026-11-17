# Prior art: pytest per-ID outcome comparison (pytest project); exact strings.
# Ours: order-specific orchestration, same source/tests for both modules.
from pathlib import Path
import sys,subprocess,os,json,shutil,hashlib,datetime,collections
A=Path('artifacts/apa_p1'); B=json.loads((A/'build.json').read_text()); dst=Path(B['module']); port=Path(B['saved_module']); prod=Path(B['production_module'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(port)==B['module_sha256']
files=sorted(str(p) for p in Path('tensor_cuda/tests').glob('test_apa_sp*.py'))+['tensor_cuda/tests/test_dlpack.py','tensor_cuda/tests/test_engine.py','tensor_cuda/tests/test_ops_phase2.py']
registration=dict(files=files,policy='All selected production files unchanged; no deselections or added skips. Native no-device failures and historical pinned-build failures are retained, not hidden. G3 compares exact per-ID outcomes; it does not claim every legacy suite passes on CPU.',source_hashes={p:sha(Path(p)) for p in files},seed=142,threads=1)
(A/'G3_CENSUS.json').write_text(json.dumps(registration,indent=2)+'\n')
env=os.environ.copy(); env.update(PYTHONPATH=str(Path('tensor_cuda').resolve())+':'+str(Path('tensor_cuda/tests').resolve())+':'+str(A.resolve()),PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',TMPDIR='/tmp/apa_p1_seat')
exits={}
try:
    for arm,source in [('production',prod),('port',port)]:
        shutil.copy2(source,dst)
        assert sha(dst)==sha(source)
        cmd=[sys.executable,str(A/'pytest_receipt.py'),str(A/f'g3_{arm}.json'),'-q','--tb=short','-ra','-p','no:cacheprovider','--basetemp',f'/tmp/apa_p1_seat/g3_{arm}',*files]
        with (A/f'g3_{arm}.log').open('w') as f:
            try: code=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=650).returncode
            except subprocess.TimeoutExpired:code=124
        exits[arm]=dict(exit=code,argv=cmd)
finally:
    shutil.copy2(port,dst)
    assert sha(dst)==B['module_sha256']
(A/'g3_exits.json').write_text(json.dumps(exits,indent=2)+'\n')
print(json.dumps(exits,indent=2))
