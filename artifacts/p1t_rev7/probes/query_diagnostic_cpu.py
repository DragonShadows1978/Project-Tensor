"""PASS iff synchronize identifies the failing cudaEventQuery and does not claim successful stream sync failed.
CPU actual-source trace only; build_cpu.sh builds its inclusion of reviewed dlpack_operand.cpp. No GPU execution.
Prior art: CUDA12.6 EventQuery/StreamSynchronize API contracts (NVIDIA2024). Taken: site-specific error attribution;
ours: success of phase1 with a one-shot query error in phase2. No prior art known to me for the tiny assertion itself.
"""
import _p1t_rev6_trace as n
n.queue(None,0);n.inject('query')
try:n.synchronize()
except RuntimeError as e:msg=str(e)
else:raise AssertionError('event-query error was lost')
print('Successful stream sync devices:',n.syncs());print(msg)
assert n.syncs()==[0]
assert 'cudaEventQuery' in msg and 'could not be synchronized' not in msg,'FAIL: successful sync misreported instead of event query'
print('PASS query error attributed correctly')
