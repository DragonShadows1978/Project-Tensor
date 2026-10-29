"""TX3-X5-REV2: direct evaluation of raw receipt operands, never import GPU/test code.
Prior art: APA SP1.1 R26 (2026), prefix-mask and dense_scores definitions in apa_sp1_reference.py.
Taken: inclusive prefix criterion, original operands, standard softmax/sink denominator.
New: only this independent transcription/evaluation, no new selector or optimization.
Online softmax: Milakov/Gimelshein (2018); FlashAttention: Dao et al. (2022).
Literature unverified -- lead to check: online normalizer calculation for softmax.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import ast
import json

receipt = Path('/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tests/test_apa_sp1_1.py')
tree = ast.parse(receipt.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and
          n.name == 'test_extra_refinement_changes_output_and_cannot_be_called_work_only')
raw = {}
for node in fn.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ('bulk', 'exact', 'values'):
            raw[node.targets[0].id] = ast.literal_eval(node.value.args[0])
call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and
            isinstance(n.func, ast.Attribute) and n.func.attr == 'partition_online')
raw['delta'] = ast.literal_eval(call.args[3])
raw['partitions'] = ast.literal_eval(call.args[4])
raw['sink'] = ast.literal_eval(next(k.value for k in call.keywords if k.arg == 'sink'))
assert raw == {'bulk':[2,0], 'exact':[2,10], 'values':[[0],[1]],
               'delta':0, 'partitions':[[0],[1]], 'sink':0}, raw
# Direct evaluation of the registered inclusive-prefix comparison, not a new algorithm.
b, x = raw['bulk'], raw['exact']
gmask = [v >= max(b[:j+1]) - raw['delta'] for j,v in enumerate(b)]
smask = [False] * len(b)
for part in raw['partitions']:
    for j in part:
        smask[j] = b[j] >= max(b[i] for i in part if i <= j) - raw['delta']
gscores = [x[j] if gmask[j] else b[j] for j in range(len(b))]
sscores = [x[j] if smask[j] else b[j] for j in range(len(b))]
with localcontext() as ctx:
    ctx.prec = 80
    values = [Decimal(row[0]) for row in raw['values']]
    def output(scores, sink):
        weights = [Decimal(s).exp() for s in scores]
        num = sum((w*v for w,v in zip(weights,values)),Decimal(0))
        den = sum(weights,Decimal(0)) + (Decimal(sink).exp() if sink is not None else Decimal(0))
        return num/den
    global_sink = output(gscores,raw['sink'])
    split_sink = output(sscores,raw['sink'])
    global_no_sink = output(gscores,None)
    split_no_sink = output(sscores,None)
    assert global_sink == 1/(Decimal(2).exp()+2)
    assert split_sink == Decimal(10).exp()/(Decimal(2).exp()+Decimal(10).exp()+1)
    assert split_sink-global_sink > Decimal('0.8')
    assert split_no_sink-global_no_sink > Decimal('0.8')
    result = {'evidence_class':'CPU Decimal illustration of exact real formulas; not a floating-point kernel test',
              'raw_operand_source':str(receipt)+':67-74','precision_decimal_digits':ctx.prec,
              'raw_operands':raw, 'global_mask':gmask,'split_mask':smask,
              'global_mixed_scores':gscores,'split_mixed_scores':sscores,
              'sink_contribution':str(Decimal(raw['sink']).exp()),
              'global_formula':'1/(exp(2)+2)', 'split_formula':'exp(10)/(exp(2)+exp(10)+1)',
              'global_with_sink':str(global_sink),'split_with_sink':str(split_sink),
              'gap_with_sink':str(split_sink-global_sink),
              'global_without_sink':str(global_no_sink),'split_without_sink':str(split_no_sink),
              'gap_without_sink':str(split_no_sink-global_no_sink)}
Path('artifacts/x5_rev2/splitk_outputs.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
