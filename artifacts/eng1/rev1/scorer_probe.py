# Prior art: GN1 A8 scorer. Run the frozen scorer against unchanged real audit data.
# Sole output redirection preserves the read-only order; evaluation code and all inputs are unmodified.
import os,sys,runpy,json
from pathlib import Path
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'; tool=root/'artifacts/eng1/tool'
os.environ['RUNS']=str(tool/'audit_final')
sys.path.insert(0,str(tool))
original=Path.write_text
def output(self,data,*a,**kw):
    if self == tool/'audit_final/G5_SUMMARY.json':
        return original(rev/'audit_rescore.json',data,*a,**kw)
    raise RuntimeError('unexpected write: '+str(self))
Path.write_text=output
try:
    runpy.run_path(str(tool/'score_gpu.py'),run_name='__main__')
except SystemExit as e:
    print('SCORER_EXIT', e.code)
    raise
