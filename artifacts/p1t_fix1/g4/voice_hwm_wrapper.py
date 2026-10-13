"""P1T-FIX1 G4 (informational): run Voice's unchanged runner in-process (runpy) and report tc_signal's deferred-release
counters at the end (the pending-list high-water mark). Voice's files are only read. Usage (cwd = Project-Voice):
    python3 voice_hwm_wrapper.py <out_json> <runner args...>
Prior art: runpy.run_path (CPython stdlib). Nothing new."""
import json
import runpy
import sys

RUNNER = 'engine/tensor_requests/p1t_tests/run_p1t.py'
out_json = sys.argv[1]
sys.argv = [RUNNER] + sys.argv[2:]
code = 0
try:
    runpy.run_path(RUNNER, run_name='__main__')
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
from tc_signal import _tc_signal as C  # noqa: E402  (already imported by the adapter)
stats = dict(C.pending_stats())
json.dump({'runner_exit': code, 'pending_stats_at_end': stats}, open(out_json, 'w'), indent=1)
print('PENDING_STATS', json.dumps(stats), flush=True)
sys.exit(code)
