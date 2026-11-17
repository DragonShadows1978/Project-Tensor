# Prior art: process-exit receipts (AB-1 2026), SHA-256 NIST 2015.
from pathlib import Path
import sys,json,datetime,hashlib
run=Path(sys.argv[1]);label=sys.argv[2];code=int(sys.argv[3]);log=run/(label+'.log')
result=dict(label=label,exit=code,finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=sys.argv[4:],log=str(log),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None,comparison=str(run/label/'comparison.json'))
with (run/(label+'.exit.json')).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
