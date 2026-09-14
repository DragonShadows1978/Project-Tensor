"""Prior art: BP-KERNEL-3 create-only SHA256 manifest, taken; BK4 protocol pins ours."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,difflib,hashlib
r=Path(__file__).resolve().parents[2];sys.path.insert(0,str(r/'scripts'))
import bp_kernel_4 as k
files=list((r/'tensor_cuda/src').glob('*.cu'))+list((r/'tensor_cuda/src').glob('*.cpp'))+list((r/'tensor_cuda/src').glob('*.cuh'))+list((r/'tensor_cuda/include').rglob('*.h'))+list((r/'tensor_cuda/tensor_cuda').rglob('*.py'))+[r/'tensor_cuda/CMakeLists.txt']
files += [r/p for p in ('scripts/bp_kernel_1.py','scripts/bp_kernel_2.py','scripts/bp_kernel_3.py','scripts/bp_kernel_4.py','scripts/bp_census_4.py','tests/test_bp_kernel_4.py','orders/BP_KERNEL_4_FORWARD_TENSOR_CORE.md')]
files += [k.INPUT,k.STATE,k.BACK_REFERENCE,k.ART/'build_driver.py',k.ART/'register.py',k.ART/'lead_commands.txt',k.ART/'baseline_regions.json']+list((k.ART/'baseline').glob('*'))
stat={};patch=[]
for name in ('kernels.cu','ops.cpp','bindings.cpp'):
 b=(k.ART/'baseline'/name).read_text().splitlines(True);a=(r/'tensor_cuda/src'/name).read_text().splitlines(True)
 diff=list(difflib.unified_diff(b,a,fromfile='baseline/'+name,tofile='tensor_cuda/src/'+name));patch+=diff
 stat[name]=dict(added=sum(x.startswith('+') and not x.startswith('+++') for x in diff),removed=sum(x.startswith('-') and not x.startswith('---') for x in diff))
with (k.ART/'source.patch').open('x') as f:f.writelines(patch)
k.create_json(k.ART/'diff_stat.json',stat)
reg=dict(k.protocol(),pins={str(p.relative_to(r)):k.sha(p) for p in sorted(set(files))},regions=json.loads((k.ART/'baseline_regions.json').read_text()),
    reference_completion='reference_receipt.json create-only after CPU computation, binds this registration and array digest',
    engine_completion='engine_build_receipt.json create-only, binds all registered engine sources and binary digest',
    seat='Codex Astra gpt-6-astra, reasoning high (order-specified)',status='Registered before CPU build/reference/tests and all GPU gates')
k.create_json(k.REG,reg)
with k.REG.with_suffix('.sha256').open('x') as f:f.write(k.sha(k.REG)+'\n')
print('REGISTRATION_SHA256='+k.sha(k.REG))
