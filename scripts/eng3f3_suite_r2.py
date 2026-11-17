"""ENG3-F3 r2 inherited G3: unchanged inherited pytest options and per-ID reporting.
Prior art: pytest hook API (2004 onward); inherited ENG3 suite harness (2026).
Taken: phase folding, suite census/options. Ours: copied-package pin and receipt destination.
Collection failures remain compared outcomes; pytest exit is recorded, never treated as a clean suite.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
MODULE = os.environ["ENG3F3_MODULE"]
PKG_PARENT = str(__import__("pathlib").Path(MODULE).parent.parent)
EXCLUDE = ["tensor_cuda/tests/test_eng3_host.py", "tensor_cuda/tests/test_terrain_render.py", "tensor_cuda/tests/test_eng3_f3.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tree_state():
    # Detect writes without invoking git. Generated receipt roots are expected.
    state = {}
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")
                   and not os.path.islink(os.path.join(base, d))
                   and os.path.relpath(os.path.join(base, d), ROOT) not in
                       ("artifacts/eng3f3", "tensor_cuda/build-eng3f3")]
        for name in files:
            p = os.path.join(base, name)
            if not os.path.islink(p):
                st = os.stat(p)
                state[os.path.relpath(p, ROOT)] = (st.st_size, st.st_mtime_ns)
    return state


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
    out = os.path.join(ROOT, "artifacts", "eng3f3", "g3_r2", label)
    os.makedirs(out, exist_ok=True)
    got = sha(MODULE)
    if got != expect:
        print("REFUSED: module", got, "!= expected", expect)
        return 3
    os.environ["PYTHONPATH"] = PKG_PARENT
    os.environ.setdefault("TMPDIR", "/tmp/eng3f3_suite_r2")
    sys.path.insert(0, PKG_PARENT)
    import tensor_cuda as tc
    assert os.path.realpath(tc._C.__file__) == os.path.realpath(MODULE), tc._C.__file__
    import pytest
    files = os.environ.get("ENG3_FILES", "tensor_cuda/tests tests").split()
    args = ["-q", "-p", "no:cacheprovider", "-rfE", "--rootdir=" + ROOT,
            "--basetemp=/tmp/eng3f3_suite_r2/pt/" + label, "--junitxml=" + os.path.join(out, "junit_r2.xml"),
            "-o", "junit_logging=all", "--continue-on-collection-errors"]
    if not os.environ.get("ENG3_OWN_TESTS"):  # G1/G1b/G2 and mutant runs: test_eng3_host.py itself (ENG3_FILES)
        for e in EXCLUDE:
            args += ["--ignore=" + os.path.join(ROOT, e)]
    if os.environ.get("ENG3_RECORD"):
        os.environ["ENG3_RECORD_DIR"] = out
    if os.environ.get("ENG3_K"):
        args += ["-k", os.environ["ENG3_K"]]
    args += [os.path.join(ROOT, f) for f in files]
    before = tree_state()
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    plugin = Reports()
    os.chdir(ROOT)
    rc = int(pytest.main(args, plugins=[plugin]))
    after = tree_state()
    counts = {}
    for v in plugin.ids.values():
        counts[v] = counts.get(v, 0) + 1
    res = {"label": label, "start": t0, "end": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "module": MODULE, "module_sha256": got, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "pytest_args": args, "pytest_exit": rc, "counts": counts, "n_ids": len(plugin.ids),
           "collection_errors": plugin.collection_errors, "outcomes": plugin.ids, "failures": plugin.longrepr,
           "tree_state_before": before, "tree_state_after": after,
           "worktree_writes_by_tests": sorted(k for k in set(after) | set(before) if before.get(k) != after.get(k))}
    with open(os.path.join(out, "outcomes_r2.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("ENG3 suite", label, "exit", rc, counts, "collection errors", len(plugin.collection_errors),
          "new worktree writes", res["worktree_writes_by_tests"])
    return 0  # the suite's own exit is recorded; outcomes are compared per ID, not by exit code


if __name__ == "__main__":
    sys.exit(main())
