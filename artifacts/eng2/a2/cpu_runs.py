#!/usr/bin/env python3
"""ENG2 G2 CPU runs (CUDA hidden): each named arm's tests/test_eng2.py `-k cpu` subset in a fresh child interpreter with
the arm package preloaded (the staged child's method: arm_origin.bound_environment + sys.path), one thread for BLAS
(L25). For mutant arms the staged tool's MutantTrace (tool/run_gpu_tests.py) records which tests executed the
registered hunk (L41). Writes artifacts/eng2/g2/<arm>_cpu.json and <arm>_cpu.xml.
Usage: CUDA_VISIBLE_DEVICES= cpu_runs.py ARM [ARM ...]
Prior art: pytest report hooks (as in ENG1/GN1's run_gpu_tests.py); CPython sys.settrace. No new algorithm.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

G2 = Path(__file__).resolve().parent
TOOL = G2.parent / "tool"
ROOT = G2.parents[2]
CHILD = r'''
import json, os, sys
from pathlib import Path
package, arm, out, xml, tool = sys.argv[1:6]
sys.path.insert(0, package)
import tensor_cuda as tc
assert Path(tc.__file__).resolve().parent.parent == Path(package).resolve(), tc.__file__
sys.path.insert(1, tool)
import pytest
from run_gpu_tests import MutantTrace, mutant_hunk
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
class R:
    outcomes, details, ids = {}, {}, []
    def pytest_collection_finish(self, session): R.ids = [i.nodeid for i in session.items]
    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.failed:
            if report.failed or report.nodeid not in R.outcomes:
                R.outcomes[report.nodeid] = report.outcome if report.when == "call" else ("error" if report.failed else report.outcome)
            if report.failed:
                R.details[report.nodeid] = str(report.longrepr)[-600:]
plugins = [R()]
tracer = None
mutants = {r["id"] for r in json.loads((Path(tool).parent / "mutants" / "mutants.json").read_text())}
if arm in mutants:
    tracer = MutantTrace(*mutant_hunk(arm, Path(package)))
    plugins.append(tracer)
rc = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "-k", "cpu",
                      "--junitxml=" + xml, str(Path(package) / "tests" / "test_eng2.py")], plugins=plugins))
sys.settrace(None)
norm = lambda i: "tensor_cuda/tests/" + i.split("/tests/", 1)[1]
Path(out).write_text(json.dumps(dict(arm=arm, rc=rc, n=len(R.ids), tc_file=str(Path(tc.__file__).resolve()),
    engine=str(Path(tc._C.__file__).resolve()), outcomes={norm(k): v for k, v in R.outcomes.items()},
    failure_details={norm(k): v for k, v in R.details.items()},
    mutant_trace=None if tracer is None else {norm(k): sorted(v) for k, v in tracer.hits.items()}), indent=1) + "\n")
'''


def main(arms):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    sys.path.insert(0, str(TOOL))
    from arm_origin import bound_environment
    for arm in arms:
        package = TOOL / "arms" / arm / "tensor_cuda"
        env = bound_environment(package, dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                                              OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        out, xml = G2 / f"{arm}_cpu.json", G2 / f"{arm}_cpu.xml"
        p = subprocess.run([sys.executable, "-B", "-c", CHILD, str(package), arm, str(out), str(xml), str(TOOL)],
                           env=env, cwd=os.environ["TMPDIR"], capture_output=True, text=True, timeout=500)
        data = json.loads(out.read_text())
        failed = sorted(k for k, v in data["outcomes"].items() if v != "passed")
        print(json.dumps(dict(arm=arm, child_exit=p.returncode, pytest_rc=data["rc"], collected=data["n"],
                              failed=len(failed), traced=None if data["mutant_trace"] is None else len(data["mutant_trace"]))))


if __name__ == "__main__":
    main(sys.argv[1:])
