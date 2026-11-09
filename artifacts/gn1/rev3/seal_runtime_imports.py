#!/usr/bin/env python3
"""GN1-REV3 Q6: modules the sealed child/suites import at RUN time that A3's
collection-only import trace (a3/trace_imports.py, --collect-only) cannot see.
Loads numpy as collection does, then performs the two runtime accesses the
sealed code makes (run_gpu_tests.py:96 np.random.seed; test_hy3d_engine_ops.py
np.testing.*) and lists newly loaded module files absent from the seal.
Prior art: Python sys.modules origin census (PSF docs); no other prior art known to me.
"""
import hashlib, json, os, sys
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
A3 = OUT.parent / "a3"
seal = json.loads((A3 / "staging_seal.json").read_text())
trace = json.loads((A3 / "import_trace.json").read_text())
home = str(Path.home())
pinned = {p.replace("$HOME/", home + "/", 1) for p in seal["external_modules"]}
traced = {r["path"].replace("$HOME/", home + "/", 1) for r in trace["modules"]}
import numpy as np  # noqa: E402
import pytest  # noqa: E402,F401
def files():
    return {str(Path(m.__file__).resolve()) for m in list(sys.modules.values()) if getattr(m, "__file__", None) and Path(m.__file__).is_file()}
before = files()
np.random.seed(571)
_ = np.testing.assert_allclose
after = files()
new = sorted(after - before)
unpinned = [p for p in new if p not in pinned]
result = dict(numpy=np.__version__, newly_loaded_at_runtime=len(new), unpinned=len(unpinned),
              unpinned_sample=[p.replace(home, "$HOME", 1) for p in unpinned[:15]],
              in_trace=sum(p in traced for p in new), seal_external_count=len(pinned))
(OUT / "seal_runtime_imports.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=1))
