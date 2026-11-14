"""Receipt wrapper. Prior art: subprocess/pytest process execution; no novel algorithm."""
from pathlib import Path
import subprocess, sys, os, json, time
p=Path(__file__).resolve().parent
name=sys.argv[1]; argv=sys.argv[2:]
t=time.monotonic()
r=subprocess.run(argv,capture_output=True,text=True,timeout=570)
(p/(name+'.log')).write_text(r.stdout+r.stderr)
receipt=dict(command=argv,exit_code=r.returncode,seconds=time.monotonic()-t,environment={k:os.environ.get(k) for k in ['TMPDIR','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE']})
(p/(name+'.command.json')).write_text(json.dumps(receipt,indent=2)+'\n')
with (p/'LEDGER.md').open('a') as f:f.write('- Run '+name+': '+json.dumps(receipt)+'; stdout/stderr '+name+'.log.\n')
print(r.stdout+r.stderr);print(json.dumps(receipt));sys.exit(r.returncode)
