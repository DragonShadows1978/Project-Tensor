"""Exercise the real driver's regression scoring with constructed pytest receipts.
Prior art: pytest exit/report semantics; boundary-value tests (Myers, 1979).
Taken: honest pytest outcomes; ours: this order's branch census, no new scorer.
"""
from pathlib import Path
from types import SimpleNamespace
import sys, json, copy
sys.path.insert(0,'artifacts/apa_p1')
import gpu_driver as d
A=Path('artifacts/apa_p1_a1')
build=d.load(d.ART/'build.json'); real_load=d.load
test='tensor_cuda/tests/test_apa_tiled.py'; tid=test+'::test_constructed'
def receipt(arm,ids,code,failed=False):
 return dict(exit=code,collection=ids,collection_errors=[],module={'sha256':build['production_sha256' if arm=='production' else 'module_sha256']},
             reports={name:[dict(when='setup',outcome='passed',message=''),dict(when='call',outcome='failed' if failed else 'passed',message='original failure' if failed else ''),dict(when='teardown',outcome='passed',message='')] for name in ids})
rows=[]
for label,ids_a,ids_b,ca,cb,fail_a,fail_b,want in [
 ('both_empty',[],[],5,5,False,False,True),
 ('production_empty',[],[tid],5,0,False,False,False),
 ('port_empty',[tid],[],0,5,False,False,False),
 ('ordinary_pass',[tid],[tid],0,0,False,False,True),
 ('ordinary_equal_failure',[tid],[tid],1,1,True,True,True),
 ('ordinary_outcome_difference',[tid],[tid],0,1,False,True,False),
 ('ordinary_id_difference',[tid],[tid+'2'],0,0,False,False,False),
 ('collection_process_error',[],[],2,5,False,False,False)]:
 out=A/'scorer_constructed'/label;out.mkdir(parents=True)
 exp=dict(gpu_test_files={'apa':[test],'dlpack':[]},gpu_collected_ids=ids_a)
 d.load=lambda p:exp if Path(p)==d.ART/'EXPECTED.json' else build if Path(p)==d.ART/'build.json' else real_load(p)
 d.sha=lambda p:build['production_sha256'] if 'refs/production' in str(p) else build['module_sha256']
 calls=[]
 def child(cmd,log,env=None,limit=240):
  arm='production' if Path(cmd[2]).name=='production.json' else 'port';calls.append(arm)
  obj=receipt(arm,ids_a if arm=='production' else ids_b,ca if arm=='production' else cb,fail_a if arm=='production' else fail_b)
  Path(cmd[2]).write_text(json.dumps(obj,indent=2)+'\n');Path(log).write_text('Constructed pytest receipt; no pytest launched.\n')
  return dict(exit=obj['exit'],constructed=True)
 d.child=child
 got=d.regression(SimpleNamespace(out=str(out),test_file=test,group=None))
 result=real_load(out/'comparison.json')
 ok=got==want and (calls==['production','port'] if ca in (0,1,5) else calls==['production'])
 rows.append(dict(case=label,expected_pass=want,observed_pass=got,port_ran='port' in calls,status='PASS' if ok else 'FAIL',comparison=result['status'],differences=result.get('differences')))
r=dict(status='PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL',rows=rows)
(A/'SCORER_TEST.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
raise SystemExit(0 if r['status']=='PASS' else 1)
