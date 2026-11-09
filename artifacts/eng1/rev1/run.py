# Prior art: Python subprocess (PSF); GNU timeout. Taken: completed-child exit receipts. Ours: rev1 scope.
import os, sys, json, subprocess, datetime
from pathlib import Path
r=Path(__file__).resolve().parent
label,*args=sys.argv[1:]
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (r/(label+'.stdout')).open('w') as out, (r/(label+'.stderr')).open('w') as err:
    p=subprocess.run(['nice','-n','19','timeout','--kill-after=5s','550s',*args],
                     env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                              OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'),
                     stdout=out,stderr=err)
row=dict(command=args,start=start,child_exited=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         exit=p.returncode,timeout=p.returncode in (124,137),CUDA_VISIBLE_DEVICES='',threads=1)
(r/(label+'.exit.json')).write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps(row))
print((r/(label+'.stdout')).read_text()[-1800:])
print((r/(label+'.stderr')).read_text()[-800:])
