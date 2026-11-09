#!/usr/bin/env python3
"""GN1-REV3 R2-F11 reproduction (REV2's): collect the sealed regression suite in a
mirror built ONLY from A3's seal (tool files, test files, import_modules, arms),
with the candidate arm first on sys.path as run_gpu_tests.py arranges it.
CPU only (CUDA hidden); collection imports the pinned module copy but runs no test.
Prior art: REV2 mirror method (2026); no other prior art known to me."""
import json, os, sys
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
MROOT = Path("/tmp/gn1_rev3_m/W"); MA3 = MROOT / "artifacts/gn1/a3"
sys.path.insert(0, str(MA3 / "arms/candidate")); sys.path.insert(1, str(MROOT))
import tensor_cuda  # noqa: E402,F401  (the child imports the arm before pytest.main, run_gpu_tests.py:89)
assert Path(tensor_cuda.__file__).resolve() == (MA3 / "arms/candidate/tensor_cuda/__init__.py").resolve()
import pytest  # noqa: E402
class C:
    ids = []
    def pytest_collection_finish(self, session): self.ids = [i.nodeid for i in session.items]
    errs = []
    def pytest_collectreport(self, report):
        if report.failed: self.errs.append(str(report.longrepr)[-300:])
c = C()
files = json.loads((MA3 / "nn_import_test_files.json").read_text())
rc = int(pytest.main(["-q", "--collect-only", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(MROOT),
                      "--basetemp=/tmp/gn1_rev3_tmp/collect", *[str(MROOT / f) for f in files]], plugins=[c]))
expected = (MA3 / "ids_candidate_regression.txt").read_text().splitlines()
res = dict(exit=rc, collected=len(c.ids), errors=c.errs, equals_sealed_census=sorted(c.ids) == sorted(expected),
           om135_core_in_mirror=(MROOT / "scripts/apa_om135_core.py").exists())
(Path(__file__).resolve().parent / "seal_mirror_collect.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res))
