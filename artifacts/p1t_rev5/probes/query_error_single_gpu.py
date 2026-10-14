"""PASS iff the one-shot query failure is reported exactly once, completed managers drain, and next sync succeeds.
Prior art: CUDA12.6 event-query/error contracts and glibc LD_PRELOAD; ours: recoverable query regression.
LEAD ONLY, one GPU. Compile query_once_shim.c to /tmp/p1t_rev5_query_once.so; LD_PRELOAD it before Python. timeout30.
"""
import ctypes as C
import torch,tc_signal as ts
if not torch.cuda.is_available():raise SystemExit(3)
shim=C.CDLL('/tmp/p1t_rev5_query_once.so')
ts.set_allocator(lambda s:torch.empty(s,device='cuda'))
x=torch.tensor([.5,1.5,-.5],device='cuda');y=ts.rint(x)
assert ts._C.pending_stats()['entries']>0
shim.rev5_arm_query_once()
ts._C.rint_plan(x)  # next-call drain takes the failure, no new launch
assert shim.rev5_query_failures()==1
try:ts.synchronize()
except RuntimeError as e:
 assert 'cudaErrorInvalidResourceHandle' in str(e) or 'cudaErrorInvalidValue' in str(e),e
else:raise AssertionError('one-shot query error disappeared before synchronize')
assert ts._C.pending_stats()['entries']==0
assert y.cpu().tolist()==[0.,2.,-0.]
ts.synchronize()
print('PASS: query error retained and reported once; safe completion release')
