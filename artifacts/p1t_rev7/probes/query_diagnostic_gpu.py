"""LEAD ONLY, single GPU. PASS iff an injected completion-event query failure is correctly named; unavailable ->3.
No product implementation copy; actual add-on with test hook and real pending entry.
Prior art: CUDA12.6 EventQuery vs StreamSynchronize return-code attribution, NVIDIA2024.
Fix recommendation: allocation-free discriminated failure record through SyncReport.merge; ordinary tagged records,
no specific prior art known to me. Ours: distinguish a successful stream sync from a failed event query.
"""
import torch
if not torch.cuda.is_available():raise SystemExit(3)
import tc_signal as ts
ts.set_allocator(lambda shape:torch.empty(shape,device='cuda',dtype=torch.float32))
x=torch.tensor([1.5],device='cuda');y=ts.rint(x)
ts._C._test_inject(['cuda:event_query'])
try:ts.synchronize()
except RuntimeError as e:msg=str(e)
else:raise AssertionError('query failure disappeared')
print(msg)
assert 'cudaEventQuery' in msg and 'could not be synchronized' not in msg,'FAIL query failure attributed to sync'
print('PASS query failure attributed correctly')
