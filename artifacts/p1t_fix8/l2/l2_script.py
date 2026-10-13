
import ctypes, json, sys, time
import torch
import tc_signal as ts
from tc_signal import _tc_signal as C
shim = ctypes.CDLL(sys.argv[1])
mode, n = sys.argv[2], int(sys.argv[3])
torch.zeros(1, device='cuda'); torch.cuda.synchronize()
t = time.perf_counter(); torch.cuda._sleep(50_000_000); torch.cuda.synchronize()
RATE = 50_000_000 / (time.perf_counter() - t)
calls = []
def hook():                                   # no _RELEASE_EXIT_DIAG_S here: the default delay (5 s)
    for _ in range(3):
        try:
            ts.synchronize(); calls.append(['ok', None]); return
        except RuntimeError as e:
            calls.append(['RuntimeError', str(e)])
torch.cuda._sleep(int(RATE * 0.4))            # ~0.4 s of GPU work on the legacy default stream
C._test_inject(['cuda:exit_marker_create'])   # the marker cannot be created: a watchdog diagnoses the stream sync
shim.shim_arm(n if mode == 'lock' else 0, 1 if mode == 'lock' else 0, n if mode == 'join' else 0,
              50 if mode == 'join' else 0)
C._exit_hook_runner(hook)
st = C.pending_stats()
o = (ctypes.c_longlong * 16)()
shim.shim_state(o)
keys = ['wd_created', 'locks_failed', 'notifies_dropped', 'joins_failed', 'detached', 'returned_at_detach', 'wd_ops',
        'wd_ops_after_detach', 't_stop', 't_join_ret', 't_detach', 't_last_wd_op', 'wd_returned', 'held',
        'lock_fail_left', 'join_fail_left']
print('RESULT ' + json.dumps({'calls': calls, 'shim': dict(zip(keys, list(o))),
                              'stats': {k: st[k] for k in ('exit_device_wait_watchdogs', 'exit_marker_failures',
                                                           'exit_device_wait_watchdog_lock_failures',
                                                           'exit_device_wait_watchdog_detaches')}}), flush=True)
