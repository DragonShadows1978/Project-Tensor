"""LEAD GPU ONLY: decide whether capsule-only side-stream ownership survives async launch.
Prior art: DLPack consumer ownership; CUDA events; PyTorch caching allocator stream reuse.
PASS iff all12 outputs are correct and each launch adds a retained entry; finite stress, not proof of safety.
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
    retained_before=C.pending_stats()["retained_entries"]
    C.rint_run(cap,out)           # consumed owner must remain fenced behind the queued read
    assert C.pending_stats()["retained_entries"]==retained_before+1
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
print('PASS:12 finite trials correct, every call retained; no universal race-free claim')
