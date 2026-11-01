#!/usr/bin/env python3
"""GN1-REV5: run A5's REAL cpu_checks.py (seal-verified mirror copy) with the same
single substitution as parent_wrap5.py (scratch_guard prefix gn1_rev5_ instead of
gn1_a5_). Used for the L41 witness-reach re-run in the mirror.
Prior art: REV3/REV4 wrappers (2026); no prior art known to me beyond them.
"""
import os
import runpy
import sys
from pathlib import Path

MA5 = Path(os.environ["REV5_MA5"])
sys.path.insert(0, str(MA5))
import evidence  # noqa: E402


def guard(value):
    resolved = Path(value).resolve()
    assert resolved.parent == Path("/tmp") and resolved.name.startswith("gn1_rev5_"), "TMPDIR must resolve to /tmp/gn1_rev5_*"
    return resolved


evidence.scratch_guard = guard
sys.argv = [str(MA5 / "cpu_checks.py")] + sys.argv[1:]
runpy.run_path(str(MA5 / "cpu_checks.py"), run_name="__main__")
