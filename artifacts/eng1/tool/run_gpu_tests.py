#!/usr/bin/env python3
"""Pinned-arm pytest child, invoked only by the staged supervisor (adapted from GN1 A8 run_gpu_tests.py).
Prior art: pytest (Krekel et al., 2004+, year unverified), report hooks; L45 parent-after-child receipts (GN1).
ENG1 changes (ledgered): ENG1_* environment names; the arm is a package copy plus copies of test_eng1.py and
test_terrain_render.py under arms/<arm>/tensor_cuda/ (each arm pins its own engine); suites eng1 / mutation /
control / regression; the regression suite runs the worktree's existing test files with the arm package preloaded
(GN1's method) plus the arm's own terrain file, and records the E6 write-detection snapshot around the run (L46);
ENG1_CENSUS_DIR points the E3 census rows at the attempt directory; GN1's GroupNorm-specific report fields dropped.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from runtime_evidence import atomic_json, module_census
if __name__ == "__main__":
    atomic_json(Path(os.environ["ENG1_ATTEMPT"]) / "child_started.json",
                dict(pid=os.getpid(), label=os.environ["ENG1_LINE"]))

from evidence import guard, verify_start, verify_arm, lines, read
from e6_record import snapshot

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Reports:
    def __init__(self):
        self.ids, self.reports, self.collection_errors, self.exceptions = [], [], [], {}

    def pytest_collection_finish(self, session):
        self.ids = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    def pytest_runtest_makereport(self, item, call):
        if call.excinfo is not None:
            self.exceptions[(item.nodeid, call.when)] = dict(exception_class=call.excinfo.type.__name__)

    def pytest_runtest_logreport(self, report):
        self.reports.append(dict(id=report.nodeid, when=report.when, outcome=report.outcome,
                                 detail=str(report.longrepr)[-4000:] if report.failed else "",
                                 **self.exceptions.get((report.nodeid, report.when), dict(exception_class=None))))


def main():
    runs = guard(os.environ["RUNS"])
    seal, start_record = verify_start(runs)
    label = os.environ["ENG1_LINE"]
    audit = os.environ.get("ENG1_AUDIT") == "1"
    output = Path(os.environ["ENG1_ATTEMPT"]).resolve()
    assert output.parent == runs / label and output.name.startswith("attempt-"), "child attempt guard"
    assert output.is_dir()
    arm, suite = sys.argv[1:]
    line = lines()[label]
    assert (arm, suite) == (line["arm"], line["suite"]), "child line binding"
    assert audit == start_record["audit"]
    package = verify_arm(arm, seal)  # arms/<arm>/tensor_cuda
    for file, expected in seal["registration_files"].items():
        assert sha(ROOT / file) == expected, "changed registered file " + file
    module = package / "tensor_cuda" / seal["module_name"]
    assert sha(module) == seal["engine_sha256"][arm], "changed pinned engine"
    sys.path.insert(0, str(package))
    sys.path.insert(1, str(ROOT))
    import tensor_cuda as tc
    import numpy as np
    import pytest
    assert Path(tc.__file__).resolve() == (package / "tensor_cuda" / "__init__.py").resolve(), "wrong package"
    assert Path(tc._C.__file__).resolve() == module.resolve(), "wrong engine copy imported"
    np.random.seed(571)
    terrain = package / "tests" / "test_terrain_render.py"
    if suite == "regression":
        files = [str(ROOT / f) for f in read(TOOL / "regression_files.json")]
    elif suite == "terrain":
        files = [str(terrain)]
    else:
        files = [str(package / "tests" / "test_eng1.py")]
    from device_evidence import capture
    device_evidence = capture(__file__, tc._C.__file__)
    invocation = dict(device_evidence=device_evidence, arm=arm, suite=suite, audit=audit, cwd=os.getcwd(),
                      run_root=str(runs), attempt_dir=str(output), seal_sha256=sha(TOOL / "staging_seal.json"),
                      engine_sha256=sha(module), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      files=[os.path.relpath(f, ROOT) for f in files])
    (output / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    os.environ["ENG1_CENSUS_DIR"] = str(output)
    # ENG1: the regression files import sibling modules (apa_sp1_reference, test_apa_sp4g_a3), which needs pytest's
    # default prepend import mode; the eng1 suites keep GN1's importlib mode.
    mode = "--import-mode=prepend" if suite in ("regression", "terrain") else "--import-mode=importlib"
    args = ["-q", "-p", "no:cacheprovider", mode, "-rfE",
            "--rootdir=" + str(ROOT), "--basetemp=" + str(Path(os.environ["TMPDIR"]) / (label + "_pytest")),
            "--junitxml=" + str(output / "junit.xml")]
    if audit:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "audit requires hidden CUDA"
        args += ["-k", "cpu"]
    elif suite in ("mutation", "control"):
        # ENG1: C++ mutant and control lines deselect only the E6 test (it exercises no C++; its CPU reference
        # renders cost ~5 min per line); M14, the E6 mutant, runs the whole file as suite mutation_e6.
        args += ["-k", "not test_e6_terrain_writes_nothing"]
    args += files
    e6_before = snapshot(terrain) if suite == "terrain" else None
    plugin = Reports()
    rc = int(pytest.main(args, plugins=[plugin]))
    e6 = dict(before=e6_before, after=snapshot(terrain)) if suite == "terrain" else None
    result = dict(invocation, pytest_exit=rc, collected_ids=plugin.ids, collection_errors=plugin.collection_errors,
                  reports=plugin.reports, e6_record=e6, runtime_modules=module_census(ROOT))
    atomic_json(output / "child_result.json", result)
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
