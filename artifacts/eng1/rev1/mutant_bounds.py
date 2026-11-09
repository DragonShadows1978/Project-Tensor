# Prior art: DeMillo/Lipton/Sayward (1978) mutation testing; Higham (2002) IEEE roundoff bounds.
# Taken: robust value-error lower bounds and exact behavioral witnesses. Ours: registered ENG1 inputs only.
import json,sys,importlib.util,math
from pathlib import Path
import numpy as np
root=Path.cwd();rev=root/'artifacts/eng1/rev1'
N=json.loads((rev/'numerics.json').read_text())
mus=json.loads((root/'artifacts/eng1/mutants/mutants.json').read_text())
trace={
'M01':'test_e1_refuse_empty_cpu[two_neg1]: _cpu_empty((0,)).reshape([-1,-1]) -> binding:1879 -> ops::reshape:827 -> NDArray::reshape loop:767-772.',
'M02':'test_e1_infer_empty_cpu[from0_r2_p0]: reshape([-1,4]) -> binding -> ops::reshape -> resolved[infer_at]:787; inferred 1 not 0, final count refusal.',
'M03':'test_e1_refuse_empty_cpu[zero_product]: reshape([0,-1]) -> loop -> known:779; mutation converts 0 to 1, n=0, resolved=[0,0], accepted.',
'M04':'test_e1_refuse_empty_cpu[other_negative]: reshape([-2,0]) -> loop:773-775; removed check permits product 0 and returns [-2,0].',
'M05':'test_e2_pow_grad[float16-p-1.5/-0.5]: _leaf.detach.requires_grad -> pow -> weighted sum backward -> pow_scalar lambda -> mutated if:866 false -> old half formula:872.',
'M06':'Each half test_e2_pow_grad reaches branch:866 true and mutated ew_pow exponent at ops.cpp:869; same lambda for all five p.',
'M07':'test_e2_pow_grad[float16-p-1.5/-0.5] reaches same lambda:869, but casts unweighted p*x^(p-1) before upstream multiply.',
'M08':'test_e4_empty[rms_norm-last0-cpu]: child tc._C.rms_norm -> ops:460 -> tc::rms_norm -> empty guard at kernels.cu:7796.',
'M09':'test_e4_empty[apa_blend_softmax-last0-cpu]: child binding:2538 -> tc::apa_blend_softmax -> moved division at 3861 before guard; S=0; matching C_UBSAN executes guarded code.',
'M10':'test_e4_refuse[topk-k1-last0-cpu]: Tensor.topk -> ops.cpp:975 -> topk_nd -> modified range condition at kernels.cu:5380; S=0,k=1.',
'M11':'Each test_e5_refuse_cpu[*-oh_zero]: binding op -> corresponding ops wrapper -> tc::{im2col,col2im,avgpool2d,maxpool2d} -> make_dims:144; OH=0,OW=2 accepted by mutated comparison.',
'M12':'Each test_e5_refuse_cpu[*-ow_only] follows same four paths -> make_dims:144 with OH=4,OW=-2; deleted OW predicate.',
'M13':'Each test_e5_refuse_cpu[*-trunc_gap] -> make_dims floor_div lambda:138; a=-1,b=2, remainder!=0, removed decrement changes 0 to 1 output.',
'M14':'test_e6_terrain_writes_nothing lists wo8a_ppm_pair in E6_WRITERS:627 -> nested pytest runs that unguarded CPU writer -> terrain.py:3960 registered write. Own exact copied-arm run executes it and fails at test_eng1.py:683.'
}
desc={
'M01':('Return instead of MSG_TWO; invalid shape [-1,0] returned for the zero-element witness.','exact refusal message; zero slack'),
'M02':('For n=0,target[-1,4], inferred extent error is 1; count becomes 4 instead of 0, so a required successful reshape raises.','exact success/shape/value; zero slack'),
'M03':('Required zero-product refusal absent on empty input; [0,0] returned. With n=24 it raises a wrong count-mismatch message.','exact refusal/message; zero slack'),
'M04':('Required invalid-extent refusal absent; [-2,0] returned.','exact refusal/message; zero slack'),
'M08':('Rank drops 2 to 1: actual (0,), expected (2,0). Half witness likewise rank 3 to 1 and dtype remains float16.','exact shape/dtype; zero slack'),
'M09':('UBSan child exits 1 with integer division by zero; expected child 0.','exit exactly 0; zero slack'),
'M10':('No topk range refusal for S=0,k=1; mutated empty guard returns (2,1) arrays (nonempty allocation), instead of throwing.','exact refusal/message; zero slack'),
'M11':('At OH=0,OW=2, zero output accepted instead of the positive-extent refusal; eight registered IDs including trunc_gap.','exact refusal/message; zero slack'),
'M12':('At OH=4,OW=-2, invalid width accepted with zero batch; four registered IDs.','exact refusal/message; zero slack'),
'M13':('(-1)/2 truncates to 0 instead of floor -1: OH/OW become 1 instead of 0; four required refusals disappear.','exact refusal/message; zero slack'),
'M14':('One extra source-tree .ppm, 921,615 bytes in own exact copied-arm CPU run; uall and stat maps change.','exact maps; zero slack')
}
rows=[]
for m in mus:
 mid=m['id'];row=dict(id=mid,trace=trace[mid],registered_killers=m['killers'],one_edit=True)
 if mid in desc:
    row.update(error_bound=desc[mid][0],comparison=desc[mid][1])
 elif mid in ('M05','M07'):
    vals=[]
    for a in N['power']:
      if a['dtype']=='float16' and a['p'] in (-1.5,-.5):
       v=a['variants'][mid]
       vals.append(dict(p=a['p'],**v,
                        robust_lower_bound='infinite: pre-weight half value exceeds 65504 by orders of magnitude even with 4-ulp powf error'))
    row.update(numeric_witnesses=vals,comparison='non-finite mutant versus finite float64 reference and finite registered tolerance')
 else:
    # x=7 is present in every registered half input, and no scaling is needed at p in the registration.
    vals=[]
    for dt in ('float16','bfloat16'):
     for p in (-1.5,-.5,.5,2.,3.):
      ref=p*7.**(p-1)
      rt={'float16':2.**-11+2.**-20,'bfloat16':2.**-8+2.**-20}[dt]
      atol=2.**(-25 if dt=='float16' else -134)
      e=(1+(4*2.**-23))*(1+2.**-24)**2*(1+{'float16':2.**-11,'bfloat16':2.**-8}[dt])-1
      # Mutant exact real value = x * reference = 7*reference. Reverse triangle inequality.
      lb=(6-7*e)*abs(ref)-atol
      tol=rt*abs(ref)+atol
      vals.append(dict(dtype=dt,p=p,x=7.,upstream=1.,reference=ref,error_lower_bound=lb,tolerance=tol,min_ratio=lb/tol))
    row.update(numeric_witnesses=vals,comparison='each lower bound exceeds tolerance; min ratio '+str(min(v['min_ratio'] for v in vals)))
 rows.append(row)
(rev/'mutant_bounds.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Registered mutant errors and witness reachability','','Numeric bounds are CPU reasoning/emulation, not GPU kills. Non-numeric checks have exact equality (zero slack).','','| Mutant | Error versus frozen witness tolerance | Executed path |','|---|---|---|']
for row in rows:
 if row['id'] in ('M05','M07'):
  value='Infinity versus finite tolerance on both registered float16 negative-p witnesses.'
 elif row['id']=='M06':
  v=min(row['numeric_witnesses'],key=lambda v:v['min_ratio'])
  value=f"x=7: error >= {v['error_lower_bound']:.9g}, tolerance {v['tolerance']:.9g}, minimum {v['min_ratio']:.1f}x over all 10 half IDs."
 else:value=row['error_bound']+' '+row['comparison']+'.'
 lines.append('| '+row['id']+' | '+value+' | '+row['trace'].replace('|','/')+' |')
(rev/'MUTANT_BOUNDS.md').write_text('\n'.join(lines)+'\n')
print('14 error bounds, all registered witnesses reach the edit.')
print('M06 robust minimum ratio',min(v['min_ratio'] for v in rows[5]['numeric_witnesses']))
print('M05 and M07',[r['numeric_witnesses'] for r in rows if r['id'] in ('M05','M07')])
