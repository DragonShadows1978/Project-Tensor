"""Prior art: mutation testing (DeMillo/Lipton/Sayward 1978), taken; APA mutants ours.
Unverified — lead to check authors/year. In-memory source copies only, no production edits.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util,types,json,time
r=Path(__file__).resolve().parents[2];sys.path.insert(0,str(r/'scripts'))
import bp_kernel_4 as k
p=r/'tests/test_bp_kernel_4.py';spec=importlib.util.spec_from_file_location('bk4_tests',p);t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
source=(r/'scripts/bp_kernel_4.py').read_text()
mutants=[
 ('omit_abs','ab=np.where(visible,np.abs(bulk),0.)','ab=np.where(visible,bulk,0.)'),
 ('variance_plus','/count-mean*mean,0.','/count+mean*mean,0.'),
 ('omit_z','threshold=mean+zthr*','threshold=mean+0*'),
 ('strict_selection','np.abs(bulk)>=threshold[:,None]','np.abs(bulk)>threshold[:,None]'),
 ('off_by_one_causal','np.arange(lo,hi)[:,None]+1','np.arange(lo,hi)[:,None]+2'),
 ('invert_selection','sel=visible & (np.abs(bulk)>=threshold[:,None])','sel=visible & (np.abs(bulk)<threshold[:,None])'),
 ('skip_normalization',"(p@x['v'][b,kh])/den[:,None]","(p@x['v'][b,kh])"),
 ('lse_wrong_sign','m+np.log(den)','m-np.log(den)'),
 ('loosen_forward','m:2*distance[n][m]','m:3*distance[n][m]'),
 ('loosen_flips','passes=float(diff.mean())<=.005','passes=float(diff.mean())<=.01'),
 ('ignore_downstream'," and bg['candidates']['h']['verdict']=='GREEN'",''),
]
k.create_json(k.ART/'mutation_registration.json',dict(evidence_class='author mutation test registration',before_gate=True,source_sha256=k.sha(r/'scripts/bp_kernel_4.py'),tests_sha256=k.sha(p),minimum_kill_fraction=.8,mutants=[dict(name=n,before=a,after=b) for n,a,b in mutants]))
checks=[lambda:t.test_forward_dense_oracle(True,(17,31),1.0364333894937898),lambda:t.test_forward_dense_oracle(False,(15,17),-1.),t.test_percentile_contract_is_not_order_statistic,t.test_forward_gate_and_flip_exact_boundaries,t.test_downstream_red_stops_timing]
results=[]
for name,before,after in mutants:
 assert source.count(before)==1,(name,source.count(before))
 m=types.ModuleType('bk4_mutant');m.__file__=str(r/'scripts/bp_kernel_4.py');exec(compile(source.replace(before,after),m.__file__,'exec'),m.__dict__)
 t.k=m;status='SURVIVED';detail=None
 try:
  for check in checks:check()
 except AssertionError as e:status='KILLED';detail=str(e)[:400]
 except Exception as e:status='ERROR';detail=repr(e)
 results.append(dict(name=name,status=status,detail=detail))
t.k=k
nonerror=[x for x in results if x['status']!='ERROR'];killed=sum(x['status']=='KILLED' for x in nonerror);fraction=killed/len(nonerror) if nonerror else 0
receipt=dict(evidence_class='CPU author mutation baseline, not blind verification',results=results,killed=killed,nonerror=len(nonerror),kill_fraction=fraction,passes=fraction>=.8,source_unchanged=k.sha(r/'scripts/bp_kernel_4.py')==json.loads((k.ART/'mutation_registration.json').read_text())['source_sha256'])
k.create_json(k.ART/'mutation_receipt.json',receipt)
print(f'MUTATIONS {killed}/{len(nonerror)} non-error killed; errors={len(results)-len(nonerror)}; source unchanged={receipt["source_unchanged"]}')
assert receipt['passes'] and receipt['source_unchanged']
