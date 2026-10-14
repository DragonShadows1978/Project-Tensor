"""P1T-FIX2 supplementary (NOT the reviewer's probe): a variant of artifacts/p1t_rev2/probes/cpu_native.cpp whose only
change is one inserted line, `fail_next = false;`, right after the armed synchronize_and_release_all() call returns.

Why: the reviewer armed "the next host allocation fails" in a GPU-less sandbox, where the call always allocates (its
error message), so the armed failure was always consumed by the code under test. On a GPU host a fixed build makes NO
allocation on that path; the still-armed failure then fires in the probe's own teardown (pybind11's
scoped_interpreter destructor -> finalize_interpreter -> get_local_internals; gdb backtrace in
artifacts/p1t_fix2/probe_runs/rev2_oom_sync_gpu_backtrace_after.txt) and terminates there. Disarming after the call
isolates the code under test. The original probe is run unmodified as well and reported as is.
Usage: make_disarm_variant.py <cpu_native.cpp> <out.cpp>. No prior art claimed.
"""
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
anchor = '    try { tcs::synchronize_and_release_all(); }\n'
assert src.count(anchor) == 1
out = src.replace(anchor, '    try { tcs::synchronize_and_release_all(); fail_next = false; }  // P1T-FIX2 variant: disarm\n')
Path(sys.argv[2]).write_text(out)
print('variant written:', sys.argv[2])
