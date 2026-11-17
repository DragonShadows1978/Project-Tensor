"""Synchronous CPU receipt wrapper. Prior art: subprocess/timeout process supervision; no novel algorithm."""
import datetime, json, os, subprocess, sys, pathlib, hashlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/apa_p1_rev3'
label,seconds,command=sys.argv[1],int(sys.argv[2]),sys.argv[3]
assert 0<seconds<=1200
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert os.environ['TMPDIR'].startswith('/tmp/p1_rev3_')
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
log=OUT/'receipts'/f'{label}.log'
with log.open('x') as f:
 f.write(f'START {start}\nCOMMAND {command}\n'); f.flush(); output_offset=f.tell()
 r=subprocess.run(['timeout',str(seconds)+'s','bash','-c',command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 f.write(f'\nEXIT {r.returncode}\n')
meta={'label':label,'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'exit':r.returncode,'log':str(log.relative_to(ROOT)),'sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
with (OUT/'LEDGER.jsonl').open('a') as f: f.write(json.dumps(meta)+'\n')
print(log.read_text()[output_offset:]); print(json.dumps({k:v for k,v in meta.items() if k!='command'}))
sys.exit(r.returncode)
