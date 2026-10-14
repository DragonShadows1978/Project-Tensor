"""PASS iff native entry censuses, finite callback exit/interrupt/exception cases and tagged failures pass.
Uses FIX7 test payloads via AST, bypasses only GPU skip guard; never runs GPU code.
Prior art: FIX7 finite callback and first-bytecode staging (2026), CPython3.12; no new product algorithm.
"""
from pathlib import Path
import ast,json,subprocess,sys,os
root=Path(__file__).resolve().parents[3]; out=root/'artifacts/p1t_rev8'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH='/tmp/p1t_rev8_build:/tmp/p1t_rev8_build/tc_signal')
def assignment(path,name,ns=None):
    tree=ast.parse((root/path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    return eval(compile(ast.Expression(node.value),str(path),'eval'),ns or {})
rows=[]
def run(name,script,args,check):
    r=subprocess.run(['timeout','12',sys.executable,'-c',script,*args],capture_output=True,text=True,env=env,timeout=17)
    row={'case':name,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    rows.append(row); (out/'fix7_cpu_census.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert r.returncode==0,row
    resline=[ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
    res=json.loads(resline[-1][7:]) if resline else None
    check(r,res)
    print('PASS',name,flush=True)
def exited(r,res):
    assert 0<=r.stdout.find('DELETER finished')<r.stdout.find('FINALIZING True'),r.stdout
    assert res['calls']==[],res
    assert res['after']['release_admission_closed'] and res['after']['releases_in_flight']==0,res
src='tc_signal/tests/test_fix7_findings.py'
s=assignment(src,'_EXIT_SCRIPT',{'TESTS_DIR':root/'tc_signal/tests'})
classes=assignment(src,'_KNOB_CLASSES')
for key,read in classes: run('addon_knob:'+key,s,['knob',key],exited)
for kind in ['RuntimeError','MemoryError','KeyboardInterrupt','SystemExit','Marker']:run('addon_sync_raise:'+kind,s,['sync_raises',kind],exited)
for how in ['none','raises','raises_keyboardinterrupt']:run('addon_stderr:'+how,s,['stderr',how],exited)
for mode in ['interrupt_first_bytecode','c_deleted','close_raises']:run('addon_'+mode,s,[mode,'-'],exited)
def entry(r,res):
    assert res['open_after_malformed'] and all(x=='TypeError' for x in res['malformed']) and res['calls']==[],res
    vals=res['values']; default=res['default']
    for key in ['rec','str','none','huge','np32']:
        for style in ['pos','kw']:assert vals[key+':'+style]==[True,default],res
    for style in ['pos','kw']:
        assert vals['fsub:'+style]==[False,2.5] and vals['isub:'+style]==[False,3.0],res
run('addon_native_entry',assignment(src,'_ENTRY_SCRIPT'),[],entry)
def assert_runner(d):
    assert d['open_after_return'] and d['raised']=='same' and d['closed'] and d['runner_closes']==1,d
run('addon_native_runner',assignment(src,'_RUNNER_SCRIPT'),[],lambda r,d:assert_runner(d))
core=assignment('tensor_cuda/tests/test_dlpack.py','_FIX7_CORE_ENTRY_SCRIPT').replace('import tensor_cuda as tc','import _p1t_rev7_core as C\nimport types\ntc=types.SimpleNamespace(_C=C,_DLPACK_SHUTDOWN_WAIT_S=5.0)')
run('core_native_entry_runner',core,[],entry)

