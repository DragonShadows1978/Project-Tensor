# Prior art: frozen E6 pytest instrument; independent scoped completed-child supervision in run.py.
import os,sys
from pathlib import Path
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'; package=root/'artifacts/eng1/tool/arms/candidate/tensor_cuda'
sys.path.insert(0,str(package)); import tensor_cuda,pytest
os.environ['ENG1_CENSUS_DIR']=str(rev)
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--import-mode=importlib',
    '--rootdir='+str(root),'--basetemp='+os.environ['TMPDIR']+'/e6_cpu',
    '--junitxml='+str(rev/'e6_cpu.xml'),str(package/'tests/test_eng1.py')+'::test_e6_terrain_writes_nothing']))
