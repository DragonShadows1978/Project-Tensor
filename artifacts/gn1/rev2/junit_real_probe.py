#!/usr/bin/env python3
"""GN1-REV2 Q3/Q4 honest-run check: with REAL pytest output, do A2's
summary_outcomes() and junit_outcomes() agree ID by ID, and does xml_key map
every real JUnit testcase of the regression suite? CUDA is hidden, so device
tests fail or skip on CPU: that yields many real failed/skipped/error shapes.
Uses the sealed mirror copies of evidence.py and run_gpu_tests.Reports, the
candidate arm package (engine imported, never initialised on a device).

Prior art: pytest junitxml (pytest project). Ours: the comparison.
"""
import json
import os
import sys
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
MROOT = Path("/tmp/gn1_rev2_regen/W")
MA2 = MROOT / "artifacts/gn1/a2"
sys.path.insert(0, str(MA2 / "arms/candidate"))
sys.path.insert(1, str(MROOT))
sys.path.insert(2, str(MA2))
import evidence  # noqa: E402
import run_gpu_tests  # noqa: E402
import pytest  # noqa: E402
import tensor_cuda  # noqa: E402

assert Path(tensor_cuda.__file__).resolve() == (MA2 / "arms/candidate/tensor_cuda/__init__.py").resolve()
files = json.loads((MA2 / "nn_import_test_files.json").read_text())
junit = Path("/tmp/gn1_rev2_tmp/junit_real/junit.xml")
junit.parent.mkdir(parents=True, exist_ok=True)
plugin = run_gpu_tests.Reports()
rc = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(MROOT),
                      "--basetemp=/tmp/gn1_rev2_tmp/junit_real/base", "--junitxml=" + str(junit)]
                     + [str(MROOT / f) for f in files], plugins=[plugin]))
data = dict(collected_ids=plugin.ids, reports=plugin.reports)
summ = evidence.summary_outcomes(data)
dual = {i for i in plugin.ids
        if any(r["id"] == i and r["when"] == "call" and r["outcome"] == "failed" for r in plugin.reports)
        and any(r["id"] == i and r["when"] == "teardown" and r["outcome"] == "failed" for r in plugin.reports)}
err = None
try:
    xml, _ = evidence.junit_outcomes(junit, plugin.ids, dual)
except AssertionError as exc:
    xml, err = {}, str(exc)
census = sorted(open(MA2 / "ids_candidate_regression.txt").read().splitlines())
disagree = sorted(i for i in set(summ) | set(xml) if summ.get(i) != xml.get(i))
counts = {}
for o in summ.values():
    counts[o] = counts.get(o, 0) + 1
none_class_failures = sorted({r["id"] for r in plugin.reports if r["outcome"] == "failed" and r["exception_class"] is None})
veto_ids = sorted({r["id"] for r in plugin.reports if r["outcome"] == "failed" and any(
    p.casefold() in r["detail"].casefold() for p in evidence.DEVICE_PATTERNS)})
result = dict(pytest_exit=rc, collected=len(plugin.ids), census_equal=sorted(plugin.ids) == census,
              collection_errors=len(plugin.collection_errors), outcome_counts=counts, junit_error=err,
              summary_vs_junit_disagreements=disagree[:50], n_disagree=len(disagree), dual_ids=sorted(dual),
              failures_with_no_exception_class=none_class_failures[:20], n_device_pattern_failures=len(veto_ids))
(OUT / "junit_real_probe.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=1))
