"""Report-only GPU1 zero-test re-score, invoked from the original tool directory.
Prior art: pytest exit/report API; receipt replay (P1, 2026). No new scorer:
call the fixed original driver, replay production receipt and actually run port.
"""
from pathlib import Path
from types import SimpleNamespace
import os,sys,json,hashlib,argparse,shutil
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ.setdefault('TMPDIR','/tmp/apa_p1_a1_rescore')
Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
assert Path.cwd().name=='apa_p1'
sys.path.insert(0,str(Path.cwd()))
import gpu_driver as d
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
out=Path(args.out).resolve();out.mkdir(parents=True)
source=Path('gpu_20261103T062254Z/reg_apa_tiled').resolve()
real_child=d.child
def child(cmd,log,env=None,limit=240):
 if Path(cmd[2]).name=='production.json':
  shutil.copy2(source/'production.json',cmd[2]);shutil.copy2(source/'production.log',log)
  return dict(exit=5,replayed=True,source='artifacts/apa_p1/gpu_20261103T062254Z/reg_apa_tiled/production.json',sha256=hashlib.sha256((source/'production.json').read_bytes()).hexdigest())
 return real_child(cmd,log,env=env,limit=limit)
d.child=child
ok=d.regression(SimpleNamespace(out=str(out),test_file='tensor_cuda/tests/test_apa_tiled.py',group=None))
exits=sorted(Path('gpu_20261103T062254Z').glob('*.exit.json'))
historical={p.stem:json.loads(p.read_text()) for p in exits}
summary=dict(status='PASS_REPORT_ONLY' if ok else 'FAIL_REPORT_ONLY',original_registered_verdict='RED; unchanged',reg_apa_tiled_rescore='PASS' if ok else 'FAIL',original_line_count=len(exits),original_failed_lines=[k for k,v in historical.items() if v.get('exit',v.get('exit_code',0))],port_arm_actually_run_with_cuda_empty=True,tmpdir=os.environ['TMPDIR'],predictions='No new P149-P151 evidence; this does not replace restaging.',prior_art='pytest exit code 5; original fixed P1 regression scorer')
(out/'RESCORE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2));raise SystemExit(0 if ok else 1)
