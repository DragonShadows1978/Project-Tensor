"""Create-only registration; no prior art known to me for this order's schema.
Taken: SHA-256 (NIST 2001), Python difflib; unverified — lead to check.
"""
from pathlib import Path
import sys,json,difflib,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import bp_kernel_2 as k
assert not k.REG.exists()
parent=json.loads((ROOT/'artifacts/bp_kernel_1/registration.json').read_text())
source={};patch=[];stat={}
for name in ('kernels.cu','ops.cpp','bindings.cpp','autograd.cpp'):
 p=ROOT/'tensor_cuda/src'/name;b=k.ART/'baseline'/name
 source[str(p.relative_to(ROOT))]=dict(before_sha256=k.sha(b),after_sha256=k.sha(p),baseline=str(b.relative_to(ROOT)))
 delta=list(difflib.unified_diff(b.read_text().splitlines(True),p.read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
 patch+=delta
 stat[name]=dict(added=sum(line.startswith('+') and not line.startswith('+++') for line in delta),removed=sum(line.startswith('-') and not line.startswith('---') for line in delta))
with (k.ART/'source.patch').open('x') as f:f.writelines(patch)
k.create_json(k.ART/'diff_stat.json',stat)
pin=parent['variant_a_region'];data=(ROOT/'tensor_cuda/src/kernels.cu').read_bytes()
assert hashlib.sha256(data[pin['start_byte']:pin['end_byte_exclusive']]).hexdigest()==pin['sha256']
build=json.loads((k.ART/'engine_build_receipt.json').read_text());assert build['rc']==0 and len(build['binaries'])==1
files=[k.INPUT,k.STATE,k.REFERENCE,ROOT/'scripts/bp_kernel_1.py',ROOT/'scripts/bp_kernel_2.py',ROOT/'scripts/bp_census_2.py',ROOT/'tests/test_bp_kernel_2.py',ROOT/'orders/BP_KERNEL_2_ATOMICS_FREE_DKDV.md']
files+=list((ROOT/'tensor_cuda/tensor_cuda').glob('*.py'))+list((ROOT/'tensor_cuda/src').glob('*.cpp'))+list((ROOT/'tensor_cuda/src').glob('*.cu'))+list((ROOT/'tensor_cuda/include/tc').glob('*.h'))+[ROOT/'tensor_cuda/CMakeLists.txt']
files+=[k.ART/n for n in ('reference_registration.json','reference_receipt.json','engine_build_receipt.json','timing_hook_copy.json','lead_commands.txt','source.patch','diff_stat.json','register.py','build_driver.py')]
files+=[ROOT/build['binaries'][0]['path']]
k.create_json(k.REG,dict(schema_version=1,experiment='BP-KERNEL-2',shape=k.SHAPE,
 prediction=k.PREDICTION,falsifier=k.FALSIFIER,step_prediction=k.STEP_PREDICTION,step_falsifier=k.STEP_FALSIFIER,
 tolerance_rule=k.TOLERANCE,budget=k.BUDGET,warmups=3,samples=10,interleaved_order='a b f d',
 input=dict(path=str(k.INPUT.relative_to(ROOT)),sha256=k.sha(k.INPUT)),
 forward_state=dict(path=str(k.STATE.relative_to(ROOT)),sha256=k.sha(k.STATE)),
 reference=dict(path=str(k.REFERENCE.relative_to(ROOT)),sha256=k.sha(k.REFERENCE)),
 sources=source,pins={str(p.relative_to(ROOT)):k.sha(p) for p in sorted(set(files))},
 variant_a_region=pin,engine_binary=build['binaries'][0],
 rowdot_rule='Probe f output-dot dQ against reference tolerance. If RED on dQ, use f_pass_a for all f correctness/timing and the census. No route selected by timing.',
 red_f_rule='No f micro-timing when RED. Census f runs anyway labelled TIMING-ONLY / NOT A VALID STEP.',
 implementation='Reuse c query pass, save one FP32 rowdot per query; key-owned register/shared reduction writes dK/dV once. No atomics in f. No dK/dV global scratch.',
 default='Original kernel and launcher prefix byte-identical; original default VJP body preserved. Explicit thread-local engine setter opt-in; no env dispatch.',
 budget_enforcement='Two distinct create-only cell claims; each foreground child timeout 300 s under nonblocking flock, parent retains lock through termination. Incomplete => INCONCLUSIVE. No shape/sample shortening.',
 prior_art=['FlashAttention-2 backward / Tri Dao 2023: key ownership/recomputation taken; APA selection belongs to project, integration ours.',
 'FlashAttention / Dao et al. 2022: output-dot identity and softmax VJP taken; BF16 rounding is explicitly gated.',
 'NVIDIA CUDA reductions/events, NumPy, SHA-256, POSIX flock, BP-KERNEL-1 and BP-CENSUS-1: existing infrastructure taken.',
 'Unverified — lead to check these titles/authors; no network.'],
 seat='Codex Astra gpt-6-astra, reasoning high (order-specified)',status='CPU preparation; GPU results pending lead'))
with k.REG.with_suffix('.sha256').open('x') as f:f.write(k.sha(k.REG)+'\n')
print('registration_sha256='+k.sha(k.REG))
