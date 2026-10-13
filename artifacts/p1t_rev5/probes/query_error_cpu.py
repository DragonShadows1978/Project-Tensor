"""PASS iff a one-shot event-query failure is reported by the next synchronize().
Prior art: CUDA 12.6 error-slot/query contracts; ours: recoverable query-error audit.
Actual source with CPU call replacements: a mechanism witness, not GPU validation.
"""
import _p1t_rev5_trace as n
calls=[]
n.queue(lambda:calls.append('deleted'))
n.fail_query()
n.drain()
print('AFTER FAILED QUERY',dict(n.stats()),'slot',n.slot(),flush=True)
try:n.synchronize()
except RuntimeError as e:
 print('PASS reported',e,flush=True)
else:
 print('FAIL synchronize returned success; calls',calls,flush=True)
 raise SystemExit(1)
