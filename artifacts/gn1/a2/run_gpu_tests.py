#!/usr/bin/env python3
"""Pinned-arm pytest child, invoked only by the staged supervisor.
Prior art: pytest (Krekel et al., 2004+, year unverified), report hooks;
L45 parent-after-child receipts from the GN1 order. GN1 wiring only.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from evidence import guard, verify_start, verify_arm, lines

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


class Reports:
    def __init__(self):
        self.ids = []
        self.reports = []
        self.collection_errors = []
        self.exceptions = {}

    def pytest_collection_finish(self, session):
        self.ids = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    def pytest_runtest_makereport(self, item, call):
        if call.excinfo is not None:
            exc = call.excinfo
            refusal = any(entry.frame.f_locals.get("self").__class__.__name__ == "GroupNorm"
                          and entry.name in ("forward", "__init__")
                          and Path(str(entry.path)).name == "nn.py" for entry in exc.traceback)
            self.exceptions[(item.nodeid, call.when)] = dict(exception_class=exc.type.__name__, groupnorm_refusal=refusal)

    def pytest_runtest_logreport(self, report):
        self.reports.append(dict(id=report.nodeid, when=report.when,
                                 outcome=report.outcome, detail=str(report.longrepr) if report.failed else "",
                                 **self.exceptions.get((report.nodeid, report.when), dict(exception_class=None, groupnorm_refusal=False))))


def main():
    runs = guard(os.environ["RUNS"])
    seal, start_record = verify_start(runs)
    label = os.environ["GN1_LINE"]
    audit = os.environ.get("GN1_AUDIT") == "1"
    output = Path(os.environ["GN1_ATTEMPT"]).resolve()
    assert output.parent == runs / label and output.name.startswith("attempt-"), "child attempt guard"
    assert output.is_dir()
    arm, suite = sys.argv[1:]
    line = lines()[label]
    assert (arm, suite) == (line["arm"], line["suite"]), "child line binding"
    assert audit == start_record["audit"]
    package = verify_arm(arm, seal)
    pin = json.loads((TOOL / "engine_pin.json").read_text())
    seal = json.loads((TOOL / "staging_seal.json").read_text())
    paths = seal["test_files"]
    for file, expected in paths.items():
        assert sha(ROOT / file) == expected, "changed test " + file
    for file, expected in seal["arms"][arm].items():
        assert sha(package / file) == expected, "changed arm file " + file
    module = ROOT / pin["module"]
    assert sha(module) == pin["sha256"], "changed pinned engine"
    sys.path.insert(0, str(package.parent))
    sys.path.insert(1, str(ROOT))
    import tensor_cuda as tc
    import numpy as np
    import pytest
    assert Path(tc.__file__).resolve() == (package / "__init__.py").resolve()
    assert Path(tc.nn.__file__).resolve() == (package / "nn.py").resolve()
    assert Path(tc._C.__file__).resolve() == (package / module.name).resolve(), "wrong engine copy imported"
    assert sha(package / module.name) == pin["sha256"], "engine copy differs from pin"
    np.random.seed(571)
    files = (["tensor_cuda/tests/test_groupnorm_gn1.py", "tensor_cuda/tests/test_ext_phase7.py"]
             if suite == "focus" else ["tensor_cuda/tests/test_groupnorm_gn1.py"] if suite == "mutation"
             else json.loads((TOOL / "nn_import_test_files.json").read_text()))
    invocation = dict(arm=arm, suite=suite, audit=audit, cwd=os.getcwd(),
                      run_root=str(runs), attempt_dir=str(output), seal_sha256=sha(TOOL / "staging_seal.json"),
                      engine_sha256=sha(module), nn_sha256=sha(package / "nn.py"),
                      CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"), files=files)
    (output / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    args = ["-q", "-p", "no:cacheprovider", "--import-mode=importlib",
            "--rootdir=" + str(ROOT), "--basetemp=" + str(Path(os.environ["TMPDIR"]) / (label + "_pytest")),
            "--junitxml=" + str(output / "junit.xml")]
    if audit:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "audit requires hidden CUDA"
        args += ["-k", "reference"]
    args += [str(ROOT / file) for file in files]
    plugin = Reports()
    rc = int(pytest.main(args, plugins=[plugin]))
    result = dict(invocation, pytest_exit=rc, collected_ids=plugin.ids,
                  collection_errors=plugin.collection_errors, reports=plugin.reports)
    (output / "child_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
