#!/usr/bin/env python3
"""GN1-REV4: run A4's REAL run_gpu_line.py (seal-verified mirror copy) with exactly
one substitution, as REV3's parent_wrap.py did: A4's scratch_guard accepts only
/tmp/gn1_a4_*, and this review may write only under /tmp/gn1_rev4_*. The
substitute applies A4's identical rule with the rev4 prefix. Everything else is
A4's bytes; the mirror's seal verifies before every line.
Prior art: REV3's parent_wrap.py (2026); no prior art known to me beyond it.
"""
import os
import runpy
import sys
from pathlib import Path

MA4 = Path(os.environ["REV4_MA4"])
sys.path.insert(0, str(MA4))
import evidence  # noqa: E402


def guard(value):
    resolved = Path(value).resolve()
    assert resolved.parent == Path("/tmp") and resolved.name.startswith("gn1_rev4_"), "TMPDIR must resolve to /tmp/gn1_rev4_*"
    return resolved


evidence.scratch_guard = guard
sys.argv = [str(MA4 / "run_gpu_line.py")] + sys.argv[1:]
runpy.run_path(str(MA4 / "run_gpu_line.py"), run_name="__main__")
