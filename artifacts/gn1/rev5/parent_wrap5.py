#!/usr/bin/env python3
"""GN1-REV5: run A5's REAL run_gpu_line.py (seal-verified mirror copy) with exactly
one substitution, as REV3's parent_wrap.py and REV4's parent_wrap4.py did: A5's
scratch_guard accepts only /tmp/gn1_a5_*, and this review may write only under
/tmp/gn1_rev5_*. The substitute applies A5's identical rule with the rev5 prefix.
Everything else is A5's bytes; the mirror's seal verifies before every line.
Prior art: REV3 parent_wrap.py / REV4 parent_wrap4.py (2026); no prior art known
to me beyond them.
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
sys.argv = [str(MA5 / "run_gpu_line.py")] + sys.argv[1:]
runpy.run_path(str(MA5 / "run_gpu_line.py"), run_name="__main__")
