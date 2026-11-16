"""ENG3 G3 suite runner: production's test suite (tensor_cuda/tests + tests/, at HEAD) per test ID, for one build.

  ENG3_LABEL=<label> ENG3_EXPECT=<module sha256> [ENG3_K=<-k expr>] [ENG3_FILES="a b"] \
      python3 -B -P scripts/eng3_suite.py

- Refuses (exit 3) if the installed module's sha256 differs from ENG3_EXPECT.
- Imports the worktree package first and checks it is the one pytest's tests will import; PYTHONPATH is set to the
  package parent so tests that spawn Python children import the same build.
- pytest options, identical for every build: -q -p no:cacheprovider -rfE --rootdir=<root> --basetemp=/tmp/eng3_tmp/pt/
  <label> --junitxml -o junit_logging=all --continue-on-collection-errors; excluded: tensor_cuda/tests/test_eng3_host.py (this round's own tests, run
  separately) and tensor_cuda/tests/test_terrain_render.py (L40: it writes tracked artifacts/wo*.ppm files).
- Writes artifacts/eng3/g3/<label>/outcomes.json: per test ID, the outcome of each phase (setup/call/teardown) folded
  to one of passed/failed/skipped/error/xfailed/xpassed, plus collection errors by file. Also records `git status
  --porcelain` (read-only git) before and after, so any write by a test into the worktree is listed.

Prior art: pytest's hook API (pytest_runtest_logreport, pytest_collectreport) and JUnit XML; ENG2's run_gpu_tests.py
(this repo) for the per-ID report plugin shape. No other prior art known to me.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_PARENT = os.path.join(ROOT, "tensor_cuda")
MODULE = os.path.join(PKG_PARENT, "tensor_cuda", "_tensor_cuda.cpython-312-x86_64-linux-gnu.so")
EXCLUDE = ["tensor_cuda/tests/test_eng3_host.py", "tensor_cuda/tests/test_terrain_render.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_status():
    r = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "-uall"], capture_output=True, text=True)
    skip = ("artifacts/eng3/", "tensor_cuda/build-eng3/", "logs/eng3_", "scripts/eng3_")
    return sorted(l for l in r.stdout.splitlines() if not any(s in l for s in skip))


class Reports:
    def __init__(self):
        self.ids = {}
        self.collection_errors = {}
        self.longrepr = {}

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors[report.nodeid] = str(report.longrepr)[-1500:]
        elif report.skipped:
            self.ids[report.nodeid + "::<module skipped>"] = "skipped"

    def pytest_runtest_logreport(self, report):
        cur = self.ids.get(report.nodeid)
        if hasattr(report, "wasxfail"):
            out = "xfailed" if report.skipped else "xpassed"
        elif report.when == "call":
            out = report.outcome
        elif report.failed:
            out = "error"
        elif report.skipped:
            out = "skipped"
        else:
            out = None  # a passing setup/teardown says nothing on its own
        if out is None:
            if cur is None:
                self.ids[report.nodeid] = "pending"
            return
        if cur in (None, "pending", "passed") or out == "error":
            self.ids[report.nodeid] = out
        if out in ("failed", "error"):
            self.longrepr[report.nodeid] = str(report.longrepr)[-800:]


def main():
    label = os.environ["ENG3_LABEL"]
    expect = os.environ["ENG3_EXPECT"]
    out = os.path.join(ROOT, "artifacts", "eng3", "g3", label)
    os.makedirs(out, exist_ok=True)
    got = sha(MODULE)
    if got != expect:
        print("REFUSED: module", got, "!= expected", expect)
        return 3
    os.environ["PYTHONPATH"] = PKG_PARENT
    os.environ.setdefault("TMPDIR", "/tmp/eng3_tmp")
    sys.path.insert(0, PKG_PARENT)
    import tensor_cuda as tc
    assert os.path.realpath(tc._C.__file__) == os.path.realpath(MODULE), tc._C.__file__
    import pytest
    files = os.environ.get("ENG3_FILES", "tensor_cuda/tests tests").split()
    args = ["-q", "-p", "no:cacheprovider", "-rfE", "--rootdir=" + ROOT,
            "--basetemp=/tmp/eng3_tmp/pt/" + label, "--junitxml=" + os.path.join(out, "junit.xml"),
            "-o", "junit_logging=all", "--continue-on-collection-errors"]
    if not os.environ.get("ENG3_OWN_TESTS"):  # G1/G1b/G2 and mutant runs: test_eng3_host.py itself (ENG3_FILES)
        for e in EXCLUDE:
            args += ["--ignore=" + os.path.join(ROOT, e)]
    if os.environ.get("ENG3_RECORD"):
        os.environ["ENG3_RECORD_DIR"] = out
    if os.environ.get("ENG3_K"):
        args += ["-k", os.environ["ENG3_K"]]
    args += [os.path.join(ROOT, f) for f in files]
    before = git_status()
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    plugin = Reports()
    os.chdir(ROOT)
    rc = int(pytest.main(args, plugins=[plugin]))
    after = git_status()
    counts = {}
    for v in plugin.ids.values():
        counts[v] = counts.get(v, 0) + 1
    res = {"label": label, "start": t0, "end": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "module": MODULE, "module_sha256": got, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "pytest_args": args, "pytest_exit": rc, "counts": counts, "n_ids": len(plugin.ids),
           "collection_errors": plugin.collection_errors, "outcomes": plugin.ids, "failures": plugin.longrepr,
           "git_status_before": before, "git_status_after": after,
           "worktree_writes_by_tests": sorted(set(after) - set(before))}
    with open(os.path.join(out, "outcomes.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("ENG3 suite", label, "exit", rc, counts, "collection errors", len(plugin.collection_errors),
          "new worktree writes", res["worktree_writes_by_tests"])
    return 0  # the suite's own exit is recorded; outcomes are compared per ID, not by exit code


if __name__ == "__main__":
    sys.exit(main())
