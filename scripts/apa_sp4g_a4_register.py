"""A4 registration BEFORE implementation/gates.
Prior art: SP3 (2026) immutable experiment DAGs, Make/Feldman (1979),
SHA256/NIST (2001), classical bisection. No new algorithm.
"""
from apa_sp4g_common import *

def main():
    sources = sorted(R.glob('scripts/apa_sp4g_*'))
    sources += sorted(R.glob('tensor_cuda/tests/test_apa_sp4g*.py'))
    sources = [p for p in sources if p.is_file() and '_a4_' not in p.name]
    receipts = sorted(A.glob('jobs*/*.json'))
    before = dict(source_sha256={str(p.relative_to(R)):sha(p) for p in sources},
                  receipt_sha256={str(p.relative_to(R)):sha(p) for p in receipts})
    publish(A/'a4_before.json', before)
    snapshot=A/'a4_baseline';snapshot.mkdir(exist_ok=True)
    for name in ('RESULTS.md','lead_commands.txt','cells.json'):
        with (snapshot/name).open('xb') as f:f.write((A/name).read_bytes())
    cs=[]
    def add(name,kind,deps=(),**kw):
        est={'aggregate':[1,30],'freeze':[1,15],'eq':[1,30],
             'margin':[10,90] if kw.get('S',2048)==2048 else [60,270]}.get(kind,[90,275])
        cs.append(dict(id=name,kind=kind,depends=list(deps),S=kw.pop('S',2048),
                       arm=kw.pop('arm','A'),window=kw.pop('window',0),bits=4,
                       apa_min_context=0,worker_s=285,estimate_s=est,**kw))
        return name
    gate=add('ppl_a4_D32_2048_w0','precision',['kernel512','diag_a2_fp32_A_2048_w0'],arm='D32')
    props=[add(f'diag_a4_propagation_{arm}_2048_w0','propagation',[gate],arm=arm) for arm in 'AD']
    prop=add('diag_a4_propagation_2048_w0','aggregate',props,operation='propagation')
    trials=[]
    for i in range(12):
        trials.append(add(f'trial_a4_{i:02d}','trial',[gate,prop,'ppl_capture_B_2048_w0']+trials,
                          arm='C',index=i))
    freeze=add('freeze_a4','freeze',[gate,'ppl_capture_B_2048_w0']+trials,arm='C')
    for w in range(1,4):add(f'ppl_a4_A32_2048_w{w}','precision',[gate],arm='A32',window=w)
    add('ppl_a4_A32_8192','precision',[gate],arm='A32',S=8192)
    cp=[]
    for w in range(4):
        cp.append(add(f'ppl_a4_C_2048_w{w}','ppl_capture' if w==0 else 'ppl',[gate,freeze],arm='C',window=w))
    add('ppl_a4_C_2048','aggregate',cp,arm='C',operation='ppl')
    cap8=add('ppl_a4_C_8192','ppl_capture',[gate,freeze],arm='C',S=8192)
    margins=[]
    for S,cap in ((2048,cp[0]),(8192,cap8)):
        for layer in range(5,48,6):
            margins.append(add(f'margin_a4_C_{S}_l{layer:02d}','margin',[gate,cap],
                               arm='C',S=S,layer=layer,population_rows=S-1,capture_cell=cap))
    bmargin=[f'margin_a2_B_{S}_l{layer:02d}' for S in (2048,8192) for layer in range(5,48,6)]
    eq=add('eq_a4','eq',[gate]+bmargin+margins,arm='E')
    ep=[add(f'ppl_a4_E_2048_w{w}','ppl',[gate,eq],arm='E',window=w) for w in range(4)]
    add('ppl_a4_E_2048','aggregate',ep,arm='E',operation='ppl')
    add('ppl_a4_E_8192','ppl',[gate,eq],arm='E',S=8192)
    for S in (2048,8192):add(f'decode_a4_C_{S}','decode',[gate,freeze],arm='C',S=S)
    order=R/'orders/APA_SP4G_AMENDMENT_4.md'
    j=dict(order=dict(path=str(order.relative_to(R)),sha256=sha(order)),before_sha256=sha(A/'a4_before.json'),
      original_registration_sha256=REG_SHA,cells=cs,
      exactness=dict(cell=gate,reference='diag_a2_fp32_A_2048_w0',reference_ppl=49.92117893813879,
        tolerance=.005,rule='absolute D32 minus A32 <= 0.005 AND complete native fp32 dtype pins; else STOP all successors',
        scope='FP32 global Q/K/Kq/V and attention output; bf16 cast before o_proj, exactly A2 A32; no whole-model fp32 claim'),
      predictions=dict(lead='D32 within 0.005 of A32=49.921 (full receipt value used)',
        seat='Source inspection finds A2 already calls SP with .float() inputs and returns D32 cast bf16. No identified dtype bug. Predict rerun may repeat 53.472 and fail; assertions improve evidence, not arithmetic.'),
      protocol=dict(base='registration.json PROTOCOL-G unchanged',short='4 consecutive 2048 windows, last 1024 targets each',long='8192 prefix 0 last512',
        dtype_scope='A32 and D32 cast identical input values at APA seam; standard QK/softmax/PV or SP in fp32; shared bf16 merge/o_proj and subsequent model',
        propagation='separate A/D bf16 workers, identical immutable weights/ids/schedule; save all S-1 residual rows after 8 complete blocks including layer_scalar and after final norm; aggregate relative Frobenius vs A',
        calibration='12 bounded C trials using actual PPL window0 population, match existing B PPL capture fraction within .01; one frozen delta; no capture required for trials',
        margins='A2 original-call bitwise replay copied into A4 namespace; C capture IS window0/8192 PPL; 16 full-layer cells; empirical eq over B and C 2048/8192 only',
        E='SP2 conditional delta upward(log(100)+2*upward(max empirical eq_sp)); finite population, not universal proof',
        clean_decode='June flags, Model.decode unchanged, no diagnostic wrappers/counters or extra host copies'),
      rails=dict(worker_s=285,worker_hard_s=290,outer_s=585,outer_hard_s=588,lease_wait_s=20,cooldown_s=30,
        no_rows_ge=16384,no_retry=True,disk_gib=12,propagation_split='two loads in separate leases to keep each worker <=285s',
        load_estimate_s=[75,130],load_evidence='a2/a3 observed approximately75-126s; new-cell estimates not measured'),
      cpu=dict(dtype_pin_test='test_a4_fp32_native_dispatch_dtype_pin_and_returned_arm',
        baseline='all four SP4G test files; no skips',mutation_threshold=.8,
        mutation_sites=['cast_dropped','input_dtype_ignored','output_dtype_ignored','return_wrong_arm','gate_relaxed','gate_nan','coverage_gap','source_fingerprint_ignored']),
      prior_art=registration()['prior_art'],
      new_work='Native dtype assertions, residual capture, additive reference ruling and dispatch plumbing. No new algorithm; no prior art known to me for a distinct novel method.',
      process_safety='No git/subagents/background waits/kills. Existing execution files/receipts and large A2 payloads preserved.',
      seat=dict(model='gpt-6-astra',reasoning_effort='xhigh',evidence='logs/apa_sp4g_a4_r1.log'))
    publish(A/'amendment_013_a4_registration.json',j)
    print(sha(A/'amendment_013_a4_registration.json'))

if __name__=='__main__':main()
