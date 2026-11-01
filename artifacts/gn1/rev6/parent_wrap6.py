#!/usr/bin/env python3
"""GN1-REV6: run A8's REAL run_gpu_line.py (seal-verified mirror copy) with exactly
one substitution, as REV3's parent_wrap.py and REV4's parent_wrap4.py did: A8's
scratch_guard accepts only /tmp/gn1_a8_*, and this review may write only under
/tmp/gn1_rev6_*. The substitute applies A8's identical rule with the rev6 prefix.
Everything else is A8's bytes; the mirror's seal verifies before every line.
Prior art: REV3 parent_wrap.py / REV4 parent_wrap4.py / REV5 parent_wrap5.py (2026); no prior art known
to me beyond them.
"""
import os
import runpy
import sys
from pathlib import Path

MA8 = Path(os.environ["REV6_MA8"])
sys.path.insert(0, str(MA8))
import evidence  # noqa: E402


def guard(value):
    resolved = Path(value).resolve()
    assert resolved.parent == Path("/tmp") and resolved.name.startswith("gn1_rev6_"), "TMPDIR must resolve to /tmp/gn1_rev6_*"
    return resolved


evidence.scratch_guard = guard
sys.argv = [str(MA8 / "run_gpu_line.py")] + sys.argv[1:]
runpy.run_path(str(MA8 / "run_gpu_line.py"), run_name="__main__")
