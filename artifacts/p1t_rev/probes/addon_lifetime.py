"""LEAD GPU ONLY: decide whether capsule-only side-stream ownership survives async launch.
Prior art: DLPack consumer ownership; CUDA events; PyTorch caching allocator stream reuse.
No custom kernels. Finite stress probe: no observed reuse => INCONCLUSIVE, never proof of safety.
Run under timeout 60 with the reviewed addon on PYTHONPATH. Memory below 32 MiB; sequential trials.
"""
import gc, json
import torch, tc_signal as ts
from tc_signal import _tc_signal as C
assert torch.cuda.is_available()
ts.set_allocator(lambda shape: torch.empty(shape,device='cuda'))
S=torch.cuda.Stream()
n=1<<18
out=torch.empty(n,device='cuda')
# Warm up allocation and launches before inserting the delay.
x=torch.full((n,),1.25,device='cuda'); ts.rint(x,out=out); torch.cuda.synchronize()
results=[]
for trial in range(12):
    with torch.cuda.stream(S):
        producer=torch.full((n,),1.25,device='cuda')
    address=producer.data_ptr()
    cap,shape=C.rint_plan(producer) # producer -> legacy fence (the protocol's forward direction)
    del producer                  # managed tensor now owns the last allocator reference
    gc.collect()
    torch.cuda._sleep(20_000_000)  # pending work on legacy before add-on read
    C.rint_run(cap,out)           # reviewed code releases last DLPack owner BEFORE read
    with torch.cuda.stream(S):
        replacement=torch.empty(n,device='cuda')
        reused=replacement.data_ptr()==address
        replacement.fill_(9)
    torch.cuda.synchronize()
    ok=bool(torch.all(out==1).item())
    results.append(dict(trial=trial,reused=reused,correct=ok))
    del replacement,cap
print(json.dumps(results,indent=2))
assert all(row['correct'] for row in results), 'F01: add-on read reclaimed/reused storage'
print('No failure observed; reuse and race coverage are limited. Ownership defect remains source-established.')
