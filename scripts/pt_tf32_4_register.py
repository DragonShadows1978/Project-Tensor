#!/usr/bin/env python3
"""Create-only PT-TF32-4 registration, before implementation/gate comparisons.

Prior art: Gumbel (1958), David & Nagaraja (2003), Gaussian maxima;
NVIDIA TF32 (2020)/CUDA 12.6 (2024), Higham (2002) rounding/interval bounds;
PT-TF32-1/2/3 (2026), SHA256 (NIST 2001), immutable receipts. Taken methods;
ours: explicit peak-scale envelope, family tail risk, edge interval placement.
This file never loads measured candidate errors or slot-11 numerical results.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/pt_tf32_4'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    ART.mkdir(exist_ok=False)
    prior = ROOT / 'artifacts/pt_tf32_3/REGISTRATION.json'
    assert sha(prior) == prior.with_suffix('.sha256').read_text().strip()
    r = json.loads(prior.read_text())
    r.update(schema=4, order='orders/PT_TF32_4_FINAL.md',
             order_sha256=sha(ROOT / 'orders/PT_TF32_4_FINAL.md'),
             prior_registration=str(prior.relative_to(ROOT)), prior_sha256=sha(prior),
             phase='Immutable before PT-TF32-4 implementation, CPU tests and historical re-scoring. Amendments separate.')
    r.pop('pt_tf32_3_work')
    r['gemm_gate'].pop('min_speedup')
    r['gemm_gate'].update(required_flags=0x40202, compute_type=77,
        pass_rule='All 9 shapes x 3 directions x 2 allocation policies: actual dispatch HMMA + INPUT_TF32 + ACCUMULATOR_32F, compute 77, correct geometry, accuracy relative_L2 <= 1e-3.',
        speedup='Diagnostic only. The old 5x peak-ratio estimate was not a dispatch measurement; historical RED receipts stay RED.',
        mathmode='Removed attribute 8 is not queried. NUMERICAL_IMPL_FLAGS status, bytes and flags are mandatory; no inferred deprecated-query success.')
    model = r['rounding_model']
    model['absolute_metric'] = 'Only dK changes: max_abs/max(abs(reference)) uses the registered tail envelope. Other arrays keep their prior max bound; zero references require exact zero.'
    tail = dict(names=['dK'], family_comparisons=12, family_alpha=0.001,
        sigma_formula='epsilon = sqrt(6)*2^-11/sqrt(3) + sqrt(max(L,S,D,VD))*2^-24; sigma_envelope_abs = epsilon * max(abs(reference))',
        normalization='Homoscedastic sigma would be epsilon*RMS(reference), hence epsilon*RMS/peak after normalization. Causal dK sums are heteroscedastic. We instead ASSUME every absolute error has centered Gaussian/sub-Gaussian scale <= epsilon*reference_peak. This peak-scale envelope is an additional engineering assumption, NOT implied by rel-L2 alone and NOT a worst-case bound.',
        expected_max_formula='Leading-order conservative envelope epsilon*sqrt(2*ln(N)); actual expected max need not attain it. N=B*KVH*S*D.',
        bound_formula='epsilon*sqrt(2*ln(2*N*12/0.001)); equivalently leading envelope plus the stated positive margin',
        derivation='P(max_i |e_i| > sigma*t) <= 2*N*exp(-t^2/2); allocate alpha/12 to each array. Union bound needs no inter-element independence; sub-Gaussian marginals at the stated scale ARE assumed.',
        scope='Three arms (isolated, downstream, same_native_state) x four registered attention shapes. No selection-error allowance; L2 and selection gates unchanged.',
        stop='An exceeded tail or L2/selection bar remains RED. No refitting. Lead captures failing coordinate/native saved state for diagnosis.',
        shape_table=[])
    for case, s in enumerate(r['attention_shapes']):
        eps = math.sqrt(6)*model['scalar_rms'] + math.sqrt(max(s[k] for k in ('L','S','D','VD')))*model['fp32_unit_roundoff']
        count = s['B']*s['KVH']*s['S']*s['D']
        leading = eps*math.sqrt(2*math.log(count))
        bound = eps*math.sqrt(2*math.log(2*count*12/0.001))
        tail['shape_table'].append(dict(case=case, elements=count, expected_rel_L2=eps,
            relative_L2_bound=3*eps, normalized_max_leading=leading,
            normalized_max_margin=bound-leading, normalized_max_bound=bound))
    r['dk_tail'] = tail
    r['edge_gate'].update(
        shapes='All 40 padding/grouped cases: only dV uses the input-derived all-selected midpoint interval. No shape/coordinate exceptions. dQ/dK retain rtol=.003 atol=.00002; independent FP64 edge gates unchanged.',
        dV_model=dict(u32=2**-24, exp_ulp_slope=1.173,
            score_error='E_s=abs(scale)*gamma_(2D+1)*sum_d abs(q_d*k_d), gamma_n=n*u32/(1-n*u32)',
            log_probability_error='z=dot64(q,k)*scale-saved_native_lse; E_z=E_s+u32*(abs(z)+E_s)',
            probability_interval='[exp(z-E_z)*(1-U*2^-23), exp(z+E_z)*(1+U*2^-23)], U=2+floor(1.173*(abs(z)+E_z)); normal finite domain only. Singleton p=1, invisible p=0 exactly.',
            quantization='Outward FP32 endpoint rounding then monotone RNA_TF32. deltaP=max(abs(RNA(p_lower)-RNA(p64)),abs(RNA(p_upper)-RNA(p64))). No native candidate values used.',
            absolute_tolerance='deltaP.T @ abs(RNA_TF32(dO)) summed over grouped heads + gamma_m * (max_abs_quantized_P.T @ abs(RNA_TF32(dO))), m=(H/KVH)*ceil(L/16)*16. Include CPU FP64 evaluation roundoff. No arbitrary atol floor or relative-to-near-zero allowance.',
            rationale='The reference rounds probabilities from FP64 selected dot/exp, native from FP32 dot/__expf. A permitted midpoint crossing changes one TF32 bin, even when dV nearly cancels.',
            domain='Require kq==k, all visible pairs selected with a threshold margin, FP32 storage and normal probabilities/products. Fail closed outside this test-only model; not a general attention tolerance.'))
    names=['dispatch','gpu_units','gemm','attention_0','attention_1','attention_2','attention_3','memcheck','racecheck','synccheck']
    budgets=[15,30,60,35,35,90,90,30,60,30]
    r['slot'].update(seconds=1200, sequence=names, budgets_seconds=dict(zip(names,budgets)),
        optional_model_budgets_seconds=dict(noise_floor=120,onset=60,healthy=60,control=60,step_time=260),
        stop='One sequential lead slot, 1200-second global deadline including optional model lanes. Timeouts only stop own process group; unrun lanes BLOCKED; no background waits.')
    r['sanitizer_gate']['coverage']='All 60 inherited GPU tests; new generation-4 dispatch and interval assertions execute within them.'
    r['attribution'].update(extreme_values='Gumbel, Statistics of Extremes (1958); David & Nagaraja, Order Statistics (2003): Gaussian leading maxima. Taken union-bound tail calculation. Ours: explicit peak-scale engineering envelope and 12-array alpha allocation.',
        edge_interval='Higham (2002) gamma_n and absolute product-sum bounds; NVIDIA CUDA 12.6 (2024) __expf error 2+floor(abs(1.173*x)) ulps, TF32 (2020). Ours: propagated test-only coefficient interval.')
    r['pt_tf32_4_work']=dict(preserved='All other gates/defaults/BF16 arithmetic and disk-safe model adapters unchanged.',
        cpu='Tail/zero/finite/normalization and midpoint sign/cancellation adversarial author tests; dispatch rejects absent/wrong-shape/stale flags. Not blind verification.',
        gpu='No GPU authorized for author. Rebuild with CUDA_VISIBLE_DEVICES empty. Lead measures exact sealed build.')
    write(ART/'REGISTRATION.json',r)
    with (ART/'REGISTRATION.sha256').open('x') as f:f.write(sha(ART/'REGISTRATION.json')+'\n')
    # History seals include every receipt/source/small array. Multi-GiB legacy
    # gradient dumps are stat-pinned only, explicitly labeled, never rewritten.
    pins={}; large={}
    paths=[]
    for generation in (1,2,3):paths += list((ROOT/f'artifacts/pt_tf32_{generation}').rglob('*'))
    paths += list((ROOT/'tensor_cuda/src').glob('*')) + list((ROOT/'tensor_cuda/include/tc').glob('*'))
    paths += list((ROOT/'scripts').glob('pt_tf32*.py')) + list((ROOT/'tests').glob('*pt_tf32*'))
    paths += list((ROOT/'docs').glob('PT_TF32*')) + list((ROOT/'orders').glob('PT_TF32*'))
    paths += list((ROOT/'tensor_cuda/tensor_cuda').glob('*'))
    paths += list((ROOT/'tensor_cuda/build-tf32').rglob('kernels.cu.o'))
    for p in sorted(set(paths)):
        if not p.is_file():continue
        name=str(p.relative_to(ROOT)); st=p.stat()
        if st.st_size > 64*1024**2:large[name]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
        else:pins[name]=sha(p)
    write(ART/'BASELINE_SHA256.json',dict(pins=pins,large_stat_only=large))
    base=ART/'baseline';base.mkdir()
    for name in ('tensor_cuda/src/tf32_gemm.cu','tensor_cuda/src/attention_tf32.cu',
                 'tests/test_pt_tf32_gpu.py','tests/test_pt_tf32_3_gpu.py'):
        (base/Path(name).name).write_bytes((ROOT/name).read_bytes())
    print('REGISTERED',sha(ART/'REGISTRATION.json'))
    for row in tail['shape_table']:print('DERIVED_BEFORE_COMPARISON',json.dumps(row))
    print('BASELINE_HASHED',len(pins),'LARGE_STAT_ONLY',len(large))


if __name__=='__main__':main()
