"""Create-only preparation. Prior art: SHA-256 NIST (2001), difflib/Python,
BP-KERNEL-2 (2026), taken. Ours: this experiment's pins and diff receipt.
Unverified — lead to check those names. No GPU/network/git.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import difflib
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import bp_kernel_3 as k
r=k.ROOT;a=k.ART
if k.REG.exists(): raise FileExistsError(k.REG)
sources={};stats={};patch=[]
for name in ('kernels.cu','ops.cpp','bindings.cpp'):
    p=r/'tensor_cuda/src'/name;b=a/'baseline'/name
    before=b.read_text().splitlines(True);after=p.read_text().splitlines(True)
    edits=list(difflib.unified_diff(before,after,fromfile=str(b.relative_to(r)),tofile=str(p.relative_to(r))))
    patch.extend(edits)
    stats[str(p.relative_to(r))]=dict(added=sum(x.startswith('+') and not x.startswith('+++') for x in edits),removed=sum(x.startswith('-') and not x.startswith('---') for x in edits))
    sources[str(p.relative_to(r))]=dict(baseline=str(b.relative_to(r)),before_sha256=k.sha(b),after_sha256=k.sha(p))
with (a/'source.patch').open('x') as f:f.write(''.join(patch))
k.create_json(a/'diff_stat.json',dict(evidence_class='difflib source comparison, no git',sources=stats,new_files={p:len((r/p).read_text().splitlines()) for p in ('scripts/bp_kernel_3.py','scripts/bp_census_3.py','tests/test_bp_kernel_3.py')}))
with (a/'lead_commands.txt').open('x') as f:
    f.write('PYTHONDONTWRITEBYTECODE=1 python /mnt/ForgeRealm/wt/pt-bk3/scripts/bp_kernel_3.py\n')
    f.write('PYTHONDONTWRITEBYTECODE=1 python /mnt/ForgeRealm/wt/pt-bk3/scripts/bp_census_3.py\n')
base=json.loads((r/'artifacts/bp_kernel_2/registration.json').read_text())
for p in (k.INPUT,k.STATE,k.REFERENCE):
    if k.sha(p)!=base['pins'][str(p.relative_to(r))]:raise ValueError('parent data drift')
pins={}
paths=[k.INPUT,k.STATE,k.REFERENCE,r/'orders/BP_KERNEL_3_TILED_TENSOR_CORE.md',
    r/'scripts/bp_kernel_1.py',r/'scripts/bp_kernel_2.py',r/'scripts/bp_kernel_3.py',r/'scripts/bp_census_3.py',
    r/'tests/test_bp_kernel_3.py',r/'tensor_cuda/CMakeLists.txt',
    *list((r/'tensor_cuda/src').glob('*')),*list((r/'tensor_cuda/include').rglob('*.h')),
    *list((r/'tensor_cuda/tensor_cuda').rglob('*.py')),
    *[a/p for p in ('build_driver.py','build.log','engine_build_receipt.json','source.patch','diff_stat.json','lead_commands.txt','register.py')]]
for p in paths:
    if p.is_file():pins[str(p.relative_to(r))]=k.sha(p)
build=json.loads((a/'engine_build_receipt.json').read_text())
if build['rc']!=0 or len(build['binaries'])!=1: raise ValueError('build incomplete')
binary=build['binaries'][0];pins[binary['path']]=binary['sha256']
reg=dict(k.protocol(),pins=pins,sources=sources,engine_binary=binary,variant_a_region=base['variant_a_region'],
    input=dict(path=str(k.INPUT.relative_to(r)),sha256=k.sha(k.INPUT)),
    forward_state=dict(path=str(k.STATE.relative_to(r)),sha256=k.sha(k.STATE)),
    reference=dict(path=str(k.REFERENCE.relative_to(r)),sha256=k.sha(k.REFERENCE)),
    status='CPU preparation only; no native correctness or performance claimed',
    seat='Codex Astra gpt-6-astra, reasoning high (order-specified)',
    prior_art=['FlashAttention-2 / Dao 2023: ownership, tiling and backward structure taken; APA selective tile integration ours.',
               'NVIDIA CUDA WMMA 2017 / BF16 2020 and CUDA events 2007+: taken fragment API and timing.',
               'FlashAttention 2022 output-dot VJP, BP-KERNEL-2 reference/gate and supervisor, BP-CENSUS-1 model driver: taken.',
               'Unverified — lead to check these titles/authors; no network.'])
k.create_json(k.REG,reg)
with k.REG.with_suffix('.sha256').open('x') as f:f.write(k.sha(k.REG)+'\n')
print(f'{k.REG} sha256={k.sha(k.REG)}')
