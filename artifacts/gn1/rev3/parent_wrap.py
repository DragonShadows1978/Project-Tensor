#!/usr/bin/env python3
"""GN1-REV3: run A3's REAL run_gpu_line.py (sealed mirror copy) with exactly one
substitution: A3's scratch_guard accepts only /tmp/gn1_a3_*, and this review may
write only under /tmp/gn1_rev3_*. The substitute applies A3's identical rule
(resolved parent == /tmp, name prefix) with the rev3 prefix. Everything else is
A3's bytes; the mirror's seal verifies before every line.
Prior art: none beyond A3's own guard (no prior art known to me for the wrapper).
"""
import os
import runpy
import sys
from pathlib import Path

MA3 = Path(os.environ["REV3_MA3"])
sys.path.insert(0, str(MA3))
import evidence  # noqa: E402


def guard(value):
    resolved = Path(value).resolve()
    assert resolved.parent == Path("/tmp") and resolved.name.startswith("gn1_rev3_"), "TMPDIR must resolve to /tmp/gn1_rev3_*"
    return resolved


evidence.scratch_guard = guard
sys.argv = [str(MA3 / "run_gpu_line.py")] + sys.argv[1:]
runpy.run_path(str(MA3 / "run_gpu_line.py"), run_name="__main__")
