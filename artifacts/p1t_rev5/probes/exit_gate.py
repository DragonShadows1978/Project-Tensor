"""PASS iff child exits0 and finite deleter finishes before FINALIZING True; FAIL abort/hang/missing markers.
Prior art: CPython3.12/2023 ordinary-finalization lifecycle; ours: enforce the exit witness result in a parent.
Run timeout30 python exit_gate.py [full|trace|gpu] [--control]. GPU mode LEAD ONLY; no GPU execution in full/trace.
"""
import os,subprocess,sys
from pathlib import Path
mode=sys.argv[1] if len(sys.argv)>1 else 'full'
name={'full':'addon_exit_full_cpu.py','trace':'addon_exit_cpu.py','gpu':'addon_exit_gpu.py'}[mode]
args=[a for a in sys.argv[2:] if a=='--control']
r=subprocess.run(['timeout','20',sys.executable,str(Path(__file__).with_name(name)),*args],capture_output=True,text=True)
print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
if r.returncode==3:raise SystemExit(3)
done=r.stdout.find('DELETER finished');fin=r.stdout.find('FINALIZING True')
assert r.returncode==0 and 0<=done<fin, f'FAIL: rc={r.returncode}, done={done}, finalizing={fin}'
print('PASS: finite callback finished before ordinary finalization')
