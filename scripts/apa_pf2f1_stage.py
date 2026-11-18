"""Concrete GPU splits and immutable line pins, CPU only.
Prior art: PF2 (2026) line registration and SHA admission, inherited;
bounded greedy batching is ordinary scheduling, no prior art known to me
for this exact conservative work estimate; unverified - lead to check
"bounded batching kernel work budget". No numeric gate is narrowed.
"""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import apa_pf2f1_common as C
import apa_pf2f1_gpu as G
import apa_pf2f1_reference as R


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lines={};batch=[];work=0
    def flush():
        nonlocal batch,work
        if batch:
            name=f'G1_{sum(k.startswith("G1_") for k in lines):03d}'
            lines[name]=dict(mode='identity',arms=['PF','F1'],indices=batch,
                             estimated_scalar_work=work,args=['identity','--line',name])
            batch=[];work=0
    for idx,c in enumerate(R.g1_cases()):
        w=2*c['B']*(2*c['G'])*c['L']*(c['S']*(c['D']+c['VD'])+c['D']**2)
        # At most 128 cases or four trillion estimated scalar operations (two arms)
        # per lease. Larger full-call cases get their own line; never L=1 substitutes.
        if batch and (len(batch)>=128 or work+w>4e12):flush()
        batch.append(idx);work+=w
    flush()
    assert sorted(i for s in lines.values() for i in s['indices'])==list(range(4848))
    for kind in ('sink','norm'):
        lines[f'G2_{kind}']=dict(mode='nonfinite',arms=['F1'],args=['nonfinite','--arm','F1','--kind',kind,
                                      '--out',f'artifacts/apa_pf2f1/nonfinite/F1_{kind}'])
    lines['G4_coverage']=dict(mode='coverage-all',arms=['F1'],args=['coverage-all'])
    for kind,label in (('sink','pf1m01_sink_guard_positive'),('norm','pf1m02_norm_finite_only')):
        lines[f'G5_{kind}_mutant']=dict(mode='mutant',arms=['F1',label],args=['mutant','--kind',kind])
    lines['G6_refusals']=dict(mode='regression',arms=['F1'],shim=True,
        args=['regression','--kind','refusals','--line','G6_refusals'])
    for model in ('qwen3_1.7b','qwen3_4b'):
        for idx,(S,L) in enumerate(((2048,2048),(8192,8192),(32768,2048))):
            name=f'G6_memory_{model}_{S}_{L}'
            lines[name]=dict(mode='regression',arms=['F1'],shim=True,
                args=['regression','--kind','memory','--model',model,'--index',str(idx),'--line',name])
    scripts={str(p.relative_to(ROOT)):C.sha(p) for p in (ROOT/'scripts').glob('apa_pf2f1_*.py')}
    scripts['tensor_cuda/tests/test_apa_pf2_nonfinite.py']=C.sha(ROOT/'tensor_cuda/tests/test_apa_pf2_nonfinite.py')
    for name,spec in lines.items():
        minimum=500 if spec['mode']=='identity' or (spec['mode']=='regression' and spec['args'][2]=='memory') else \
                120 if spec['mode']=='coverage-all' else 60 if spec['mode'] in ('mutant','regression') else 30
        spec.update(min_s=minimum,max_s=540,inner_timeout_s=500,cooldown_s=600,
                    module_sha256={a:C.build.receipt(a)['module_sha256'] for a in spec['arms']})
    C.save(C.ART/'LINES.json',lines)
    C.save(C.ART/'LINE_PINS.json',dict(scripts=scripts,registration_sha256=C.sha(C.ART/'REGISTRATION.json'),
           lines_sha256=C.sha(C.ART/'LINES.json'),shim_sha256=C.sha('/tmp/pf2f1_shim/libpf2count.so')))
    (C.ART/'LINE_IDS.txt').write_text(''.join(k+'\n' for k in lines))
    cov=G.coverage_cases()
    C.save(C.ART/'COVERAGE.json',dict(status='STAGED_NOT_EXECUTED',reachable=28,registered=28,staged=28,executed=0,passed=0,
          compiled=32,unreachable=4,unreachable_rule='CAP64/SEQ=false for both dtypes and both diagnostics values',
          combinations=[dict(index=i,**c,execution='pending lead GPU') for i,c in enumerate(cov)],
          diagnostics_false='Execute no-mask instantiation and paired diagnostic form on identical operands; output bits equal, diagnostic masks equal row-decode oracle',
          tau_source='artifacts/apa_p1pf2/a1/PF_I3_A1.json',tau_sha256=C.sha(ROOT/'artifacts/apa_p1pf2/a1/PF_I3_A1.json'),
          tau={dt:json.loads((ROOT/'artifacts/apa_p1pf2/a1/PF_I3_A1.json').read_text())[dt]['tau'] for dt in ('float32','bfloat16')}))
    print('STAGED',len(lines),'lines; G1',sum(k.startswith('G1_') for k in lines),'parts / 4848 cases; coverage 28')


if __name__=='__main__':main()
