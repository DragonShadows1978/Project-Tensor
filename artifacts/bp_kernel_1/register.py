"""Create-only CPU registration. This is not a GPU cell or a correctness gate.
Prior art: house-rule preregistration (2026); ours: pins for this census.
"""
import importlib.util,json,hashlib,difflib
from pathlib import Path
root=Path('/mnt/ForgeRealm/wt/pt-bk1');art=root/'artifacts/bp_kernel_1'
spec=importlib.util.spec_from_file_location('bk1',root/'scripts/bp_kernel_1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
baseline=json.loads((art/'baseline_pins.json').read_text())
build=json.loads((art/'engine_build_receipt.json').read_text())
if build['rc'] or len(build['binaries'])!=1:raise RuntimeError('need successful unique engine build')
arrays=m.seed_inputs(m.SHAPE);m.save_npz(art/'inputs.npz',arrays)
reg=dict(schema_version=1,experiment='BP-KERNEL-1',date='2026-09-13',
 seat='Codex Astra gpt-6-astra, reasoning high (order-specified)',
 base='main 754dc1c (order attestation; no git invoked); pre-edit files byte-compared to live Project-Tensor main source',
 shape=m.SHAPE,tolerance_rule=m.TOLERANCE,prediction=m.PREDICTION,falsifier=m.FALSIFIER,
 secondary='d ≤ 0.6 × a if the prediction holds and d is green.',
 variants={'a':'unaltered original kernel/default binding',
 'b':'FP32 dK/dV atomics into scratch then one cast to T',
 'c':'no dK/dV writes; timing only, not a gradient; same output allocation/zeroing as a',
 'd':'b plus rowdot=dO dot saved BF16 output from real forward; Pass A removed'},
 warmups=3,samples=10,interleaved_order='a b c d repeated for 3 warmup rounds then 10 measured rounds',
 correctness_order='a1, a2, b, d, compare, then all warmups and timing; RED timings ineligible',
 budget=dict(work_seconds=300,lease_seconds=590,cells=1,lock='/tmp/forge-gpu.lock'),
 budget_enforcement='nonblocking flock; create-only cell_claim; foreground child subprocess timeout 300 seconds, no retry; parent retains lock; self-owned child only',
 input={'path':'artifacts/bp_kernel_1/inputs.npz','sha256':m.sha(art/'inputs.npz'),
 'seed':20260913,'generator':'NumPy default_rng PCG64, standard_normal float32; BF16 RNE stored as exact float32',
 'kq':'bf16(k + 0.125 * normal); synthetic detached key approximation, not the real quantizer or captured model inputs',
 'arrays':{k:{'shape':list(v.shape),'storage_dtype':str(v.dtype),'device_dtype':'bfloat16'} for k,v in arrays.items()}},
 forward_state_registration={
 'status':'PENDING_LEAD_GPU_CELL',
 'reason':'No GPU authorized in dispatched seat; cannot produce real lse/thr/output on CPU.',
 'path':'artifacts/bp_kernel_1/forward_state.npz',
 'amendment_path':'artifacts/bp_kernel_1/forward_state_registration.json',
 'rule':'Within the only GPU cell, real apa_selective_fwd_train produces out/lse/thr once; create-only NPZ and sha256 amendment before any backward gate. Original registration and inputs.npz never rewritten.',
 'deviation':'Real-forward arrays cannot be included in pre-run inputs.npz here; pinned separately before gates. Explicit two-stage input pinning; lead review required, not hidden as fulfilled.'},
 engine_binary=build['binaries'][0],
 variant_a_region=baseline['variant_a_region'],
 effective_config=json.loads(Path('/mnt/ForgeRealm/wt/grapa-bp1/artifacts/bp_census_1/receipt.json').read_text())['effective_config'],
 geometry_evidence={
 'model_mla.py':'MLAConfig200: H=16, qk_nope_dim=64 + qk_rope_dim=32 => D=96; v_head_dim=64',
 'attention_mla.py':'kv_b expands all H heads; shared k_pe expanded to H => KVH=H=16; scale=1/sqrt(96); is_causal=True',
 'attention.py':'refine .15 => _norm_ppf(1-clamp(refine))=_norm_ppf(.85); thr=mean(abs(bulk))+zthr*sqrt(max(E(abs(bulk)^2)-mean^2,0)) over visible keys',
 'kernels.cu':'training real forward determines lse and thr; d uses saved BF16 O'},
 prior_art=['Dao et al., FlashAttention 2022: softmax VJP/recomputation and output-dot identity; ours is APA integration.',
 'NVIDIA CUDA atomicAdd/events, year unverified: wider accumulation and timing, no novelty claim.',
 'Key-parallel attention backward avoids dK/dV atomics (FlashAttention 2022 work partitioning); NOT implemented: c is deletion only.',
 'NumPy PCG64 2019 / ONeill PCG 2014, IEEE RNE, pytest Krekel 2004: fixture/testing infrastructure.',
 'All external attribution unverified — lead to check; no network.'],
 sources={name:dict(before_sha256=baseline[name]['sha256'],after_sha256=m.sha(root/'tensor_cuda/src'/name)) for name in ('kernels.cu','ops.cpp','bindings.cpp')},
 pins={})
paths=['scripts/bp_kernel_1.py','tests/test_bp_kernel_1.py','artifacts/bp_kernel_1/inputs.npz',
 'artifacts/bp_kernel_1/baseline_pins.json','artifacts/bp_kernel_1/engine_build_receipt.json',
 'artifacts/bp_kernel_1/lead_commands.txt',build['binaries'][0]['path']]
reg['pins']={p:m.sha(root/p) for p in paths}
for p in ['/mnt/ForgeRealm/wt/grapa-bp1/artifacts/bp_census_1/receipt.json',
 '/mnt/ForgeRealm/wt/grapa-bp1/grapa/model_mla.py','/mnt/ForgeRealm/wt/grapa-bp1/grapa/attention_mla.py',
 '/mnt/ForgeRealm/wt/grapa-bp1/grapa/attention.py','/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/quant.py']:
 reg['pins'][p]=m.sha(p)
m.create_json(art/'registration.json',reg)
with (art/'registration.sha256').open('x') as f:f.write(m.sha(art/'registration.json')+'\n')
patch=[];stats={}
for name in ('kernels.cu','ops.cpp','bindings.cpp'):
 a=(art/'baseline'/name).read_text().splitlines(True);b=(root/'tensor_cuda/src'/name).read_text().splitlines(True)
 d=list(difflib.unified_diff(a,b,fromfile='before/tensor_cuda/src/'+name,tofile='after/tensor_cuda/src/'+name))
 patch.extend(d);stats[name]={'added':sum(x.startswith('+') and not x.startswith('+++') for x in d),
 'removed':sum(x.startswith('-') and not x.startswith('---') for x in d)}
with (art/'source.patch').open('x') as f:f.writelines(patch)
m.create_json(art/'diff_stat.json',stats)
print('REGISTRATION '+m.sha(art/'registration.json'))
print(json.dumps(stats))
