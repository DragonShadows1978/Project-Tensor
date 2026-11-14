# Prior art: DeMillo/Lipton/Sayward (1978) mutation testing; frozen ENG1 M14 and E6 instruments.
# Taken: the EXACT registered Python mutant and tests. Ours: copy under rev1 to confine all real writes.
import sys,os,shutil,json,hashlib
from pathlib import Path
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'
src=root/'artifacts/eng1/tool/arms/M14'; dst=rev/'m14_arm'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('artifacts','__pycache__'))
for p in src.rglob('*'):
    if p.is_file() and 'artifacts' not in p.relative_to(src).parts and '__pycache__' not in p.parts:
        assert p.read_bytes()==(dst/p.relative_to(src)).read_bytes()
package=dst/'tensor_cuda'
sys.path.insert(0,str(package)); import tensor_cuda as tc,pytest
out=rev/'m14_probe';out.mkdir()
os.environ['ENG1_CENSUS_DIR']=str(out)
code=int(pytest.main(['-q','-p','no:cacheprovider','--import-mode=importlib','--rootdir='+str(root),
    '--basetemp='+os.environ['TMPDIR']+'/m14_cpu','--junitxml='+str(rev/'m14_cpu.xml'),
    str(package/'tests/test_eng1.py')+'::test_e6_terrain_writes_nothing']))
print('M14 exact copied arm module',hashlib.sha256(Path(tc._C.__file__).read_bytes()).hexdigest())
raise SystemExit(code)
