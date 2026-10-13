"""PASS iff the actual core native runner logs a recorded CUDA failure despite reporting OOM (parent checks stderr).
Prior art: CPython3.12 atexit/lifecycle, SQLite OOM injection. build_core_host_oom.py extracts current product sources.
"""
import _p1t_rev8_coreoom as n
import sys
n.record()
def hook():raise KeyboardInterrupt('exit interrupted')
try:
 if '--control' in sys.argv:n._dlpack_exit_hook_runner(hook)
 else:n.runner_oom(hook)
except KeyboardInterrupt:pass
print('recorded=',n._dlpack_release_stats()['unraisable_errors'],flush=True)
