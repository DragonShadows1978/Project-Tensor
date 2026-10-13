"""PASS iff the actual native exit logger emits at least the recorded failed API under reporting OOM.
Prior art: SQLite OOM injection, CUDA record-before-consume; product dlpack_operand.cpp is included, never copied.
Run with PYTHONPATH=/tmp/p1t_rev9_build. --control disables OOM.
"""
import _p1t_rev8_oom as n
import sys
n.queue(None,0); n.inject('query'); n.inject('message')
try: n.synchronize()
except MemoryError: pass
else: raise AssertionError('setup did not exercise reporting OOM')
assert n.has_unreported()
if '--control' in sys.argv: n.log_normal()
else: n.log_oom()
print('record_still_pending=',n.has_unreported(),flush=True)
# Exit with no later reader, as the runner does. Parent checks stderr for the failed API.
