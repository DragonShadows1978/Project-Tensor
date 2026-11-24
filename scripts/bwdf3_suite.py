"""BWD-F3 P3 suite runner: a copy of scripts/bwdf2_suite.py with arms B (BWD-F1's Build B receipts and before/
files) and F3 (artifacts/bwdf1/f3/build receipts and artifacts/bwdf1/f3/F3_src), series under artifacts/bwdf1/f3/g4,
this round's tests excluded too, Build F3 restored afterwards, BWDF3_SUITE_AUDIT for L64, TMPDIR /tmp/bwdf3_tmp.
L128: it writes the arm's FILES (tensor_cuda/src/kernels.cu, ops.cpp, bindings.cpp, include/tc/core.h,
tensor_cuda/functional.py, tc_signal/tc_signal/__init__.py, tc_signal/src/bindings.cpp, tc_signal/README.md) and both
modules into the worktree and restores Build F3's; scripts/bwdf3_l128.py checks the sha256s before and after.
BWD-F2's text follows.

BWD-F2 K3 suite runner: a copy of scripts/bwdf1_suite.py with arms B (BWD-F1's Build B receipts and before/ files)
and F2 (this round's receipts and artifacts/bwdf1/f2/F2_src), series under artifacts/bwdf1/f2/g4, this round's tests
excluded too, Build F2 restored afterwards, and BWDF3_SUITE_AUDIT for L64. BWD-F1's text follows.

BWD-F1 G4 suite runner (a copy of ENG4's scripts/eng4_suite.py, eng4 worktree, with these changes): production's
test suite per test ID for one build, over three roots (tensor_cuda/tests, tests/, tc_signal/tests); two modules
(tensor_cuda and tc_signal) installed per arm; this round's changed files; this round's own tests and
test_terrain_render.py excluded (L40: it writes the tracked artifacts/wo*.ppm files); a `snapshot` mode; and an
optional L64 audit hook. The CPU half runs this file directly; the GPU half runs it through
scripts/bwdf1_step_g4.py inside a lease.

  python3 -B -P scripts/bwdf1_suite.py <series> <arm B|E> [--budget <seconds>]
  python3 -B -P scripts/bwdf1_suite.py snapshot     (copy this round's final FILES to artifacts/bwdf1/E_src/)

- install(arm): the arm's FILES (B: artifacts/bwdf1/before = git show HEAD:, E: artifacts/bwdf1/E_src) and modules
  (<pkg>/build-bwdf1/<arm>/module) go into the worktree; each module sha256 must equal BUILD_<pkg>_<arm>.json (else
  exit, L103) and each installed engine source its source map. Build E is installed again afterwards.
- BWDF1_SUITE_AUDIT=<path> (L64): every path opened outside the worktree is counted with sys.addaudithook("open").
- Outcomes: artifacts/bwdf1/g4/<series>/part_<n>/outcomes.json and the merged outcomes.json, as ENG4's.

Prior art: pytest's hook API; ENG4's eng4_suite.py (this repo), copied; PEP 578 audit hooks (2019) for L64.
"""
import collections
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_PARENT = os.path.join(ROOT, "tensor_cuda")
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
MODULE = os.path.join(PKG_PARENT, "tensor_cuda", SO)
TS_SO = "_tc_signal.cpython-312-x86_64-linux-gnu.so"
TS_MODULE = os.path.join(ROOT, "tc_signal", "tc_signal", TS_SO)
FILES = ["tensor_cuda/src/kernels.cu", "tensor_cuda/src/ops.cpp", "tensor_cuda/src/bindings.cpp",
         "tensor_cuda/include/tc/core.h", "tensor_cuda/tensor_cuda/functional.py",
         "tc_signal/tc_signal/__init__.py", "tc_signal/src/bindings.cpp", "tc_signal/README.md"]
SRC = {"B": "artifacts/bwdf1/before", "F3": "artifacts/bwdf1/f3/F3_src"}
REC = {"B": "artifacts/bwdf1/build", "F3": "artifacts/bwdf1/f3/build"}
EXCLUDE = ["tensor_cuda/tests/test_bwdf1_zero_extent.py", "tensor_cuda/tests/test_bwdf1_copy_ops.py",
           "tc_signal/tests/test_bwdf1_hardening.py", "tensor_cuda/tests/test_terrain_render.py",
           "tensor_cuda/tests/test_bwdf2_fixes.py", "tensor_cuda/tests/test_bwdf3_fixes.py"]
SKIP_STATUS = ("artifacts/bwdf1/", "tensor_cuda/build-bwdf1/", "tc_signal/build-bwdf1/", "logs/bwdf1_",
               "scripts/bwdf1_", "test_bwdf1_", "orders/", "logs/bwdf2_", "scripts/bwdf2_", "test_bwdf2_", "logs/bwdf3_",
               "scripts/bwdf3_", "test_bwdf3_") + tuple(FILES)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def install(arm):
    """Copy the arm's engine files and module into the worktree; return their sha256 after the copy."""
    rec = json.load(open(os.path.join(ROOT, REC[arm], f"BUILD_tc_{arm}.json")))
    mod = os.path.join(ROOT, "tensor_cuda/build-bwdf1", arm, "module", SO)
    if sha(mod) != rec["module_sha256"]:
        raise SystemExit(f"REFUSED (L103): {mod} sha256 differs from {REC[arm]}/BUILD_tc_{arm}.json")
    trec = json.load(open(os.path.join(ROOT, REC[arm], f"BUILD_ts_{arm}.json")))
    tmod = os.path.join(ROOT, "tc_signal/build-bwdf1", arm, "module", TS_SO)
    if sha(tmod) != trec["module_sha256"]:
        raise SystemExit(f"REFUSED (L103): {tmod} sha256 differs from {REC[arm]}/BUILD_ts_{arm}.json")
    got = {}
    for f in FILES:
        src = os.path.join(ROOT, SRC[arm], f.replace("/", "__"))
        dst = os.path.join(ROOT, f)
        shutil.copyfile(src, dst)
        got[f] = sha(dst)
        assert got[f] == sha(src), f
        for r in (rec, trec):  # the engine sources must be the ones each module was built from
            if f in r["sources"]:
                assert got[f] == r["sources"][f], (arm, f)
    for m, dst, r in ((mod, MODULE, rec), (tmod, TS_MODULE, trec)):
        tmp = dst + ".bwdf1tmp"
        shutil.copyfile(m, tmp)
        os.replace(tmp, dst)  # atomic swap (each arm keeps its receipted copy in build-bwdf1)
        got[os.path.relpath(dst, ROOT)] = sha(dst)
        assert got[os.path.relpath(dst, ROOT)] == r["module_sha256"]
    return got


def snapshot():
    d = os.path.join(ROOT, SRC["F3"])
    os.makedirs(d, exist_ok=True)
    for f in FILES:
        shutil.copyfile(os.path.join(ROOT, f), os.path.join(d, f.replace("/", "__")))
        print("F3_src", f, sha(os.path.join(d, f.replace("/", "__"))))


def git_status():
    r = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "-uall"], capture_output=True, text=True)
    return sorted(l for l in r.stdout.splitlines() if not any(s in l for s in SKIP_STATUS))


class Reports:
    def __init__(self, done, budget):
        self.ids, self.collection_errors, self.longrepr = {}, {}, {}
        self.done, self.budget, self.t0 = done, budget, time.monotonic()
        self.deselected, self.stopped, self._session = 0, False, None

    def pytest_sessionstart(self, session):
        self._session = session

    def pytest_collection_modifyitems(self, session, config, items):
        if not self.done:
            return
        drop = [i for i in items if i.nodeid in self.done]
        self.deselected = len(drop)
        if drop:
            config.hook.pytest_deselected(items=drop)
        items[:] = [i for i in items if i.nodeid not in self.done]

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
            out = None
        if out is None:
            if cur is None:
                self.ids[report.nodeid] = "pending"
            return
        if cur in (None, "pending", "passed") or out == "error":
            self.ids[report.nodeid] = out
        if out in ("failed", "error"):
            self.longrepr[report.nodeid] = str(report.longrepr)[-800:]

    def pytest_runtest_logfinish(self, nodeid, location):
        if self.budget and time.monotonic() - self.t0 > self.budget and self._session is not None:
            self.stopped = True
            self._session.shouldstop = f"bwdf1 budget {self.budget}s reached"


def main():
    if sys.argv[1] == "snapshot":
        return snapshot()
    series, arm = sys.argv[1], sys.argv[2]
    budget = int(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 0
    d = os.path.join(ROOT, "artifacts/bwdf1/f3/g4", series)
    os.makedirs(d, exist_ok=True)
    prog_path = os.path.join(d, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"arm": arm, "parts": 0, "done": []}
    assert prog["arm"] == arm, (prog["arm"], arm)
    if prog.get("complete"):
        print("series", series, "already complete; nothing to run")
        return 0
    n = prog["parts"] + 1
    part = os.path.join(d, f"part_{n}")
    os.makedirs(part)  # write-once: refuses if this part exists
    before = git_status()
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    opened = collections.Counter()
    audit_path = os.environ.get("BWDF3_SUITE_AUDIT")
    try:
        installed = install(arm)
        tsp = os.path.join(ROOT, "tc_signal")
        os.environ["PYTHONPATH"] = PKG_PARENT + os.pathsep + tsp
        os.environ.setdefault("TMPDIR", "/tmp/bwdf3_tmp")
        sys.path[:0] = [PKG_PARENT, tsp]
        import tensor_cuda as tc
        assert os.path.realpath(tc._C.__file__) == os.path.realpath(MODULE), tc._C.__file__
        import pytest
        args = ["-q", "-p", "no:cacheprovider", "-rfE", "--rootdir=" + ROOT,
                "--basetemp=/tmp/bwdf3_tmp/pt/" + series + f"_{n}", "--junitxml=" + os.path.join(part, "junit.xml"),
                "-o", "junit_logging=all", "--continue-on-collection-errors"]
        for e in EXCLUDE:
            args += ["--ignore=" + os.path.join(ROOT, e)]
        args += [os.path.join(ROOT, "tensor_cuda/tests"), os.path.join(ROOT, "tests"),
                 os.path.join(ROOT, "tc_signal/tests")]
        plugin = Reports(set(prog["done"]), budget)
        os.chdir(ROOT)
        if audit_path:
            def hook(event, a):
                if event == "open" and a and isinstance(a[0], (str, bytes)):
                    p = os.path.realpath(os.fsdecode(a[0]))
                    if not p.startswith(ROOT + os.sep):
                        opened[p] += 1
            sys.addaudithook(hook)
        rc = int(pytest.main(args, plugins=[plugin]))
    finally:
        restored = install("F3")
    if audit_path:
        with open(audit_path, "w") as f:
            json.dump({"series": series, "arm": arm, "outside_worktree_opens": dict(sorted(opened.items())),
                       "distinct_paths": len(opened), "total_opens": sum(opened.values())}, f, indent=1)
    after = git_status()
    ran = dict(plugin.ids)
    counts = {}
    for v in ran.values():
        counts[v] = counts.get(v, 0) + 1
    res = {"series": series, "arm": arm, "part": n, "start": t0,
           "end": datetime.datetime.now(datetime.timezone.utc).isoformat(), "installed": installed,
           "restored_F3": restored, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "budget_s": budget, "stopped_by_budget": plugin.stopped, "deselected_already_run": plugin.deselected,
           "pytest_args": args, "pytest_exit": rc, "counts": counts, "n_ids": len(ran),
           "collection_errors": plugin.collection_errors, "outcomes": ran, "failures": plugin.longrepr,
           "git_status_before": before, "git_status_after": after,
           "worktree_writes_by_tests": sorted(set(after) - set(before))}
    with open(os.path.join(part, "outcomes.json"), "w") as f:
        json.dump(res, f, indent=1)
    prog["parts"] = n
    prog["done"] = sorted(set(prog["done"]) | {k for k in ran if not k.endswith("::<module skipped>")})
    prog["complete"] = not plugin.stopped
    with open(prog_path, "w") as f:
        json.dump(prog, f)
    merged, coll, writes = {}, {}, []
    for i in range(1, n + 1):
        o = json.load(open(os.path.join(d, f"part_{i}", "outcomes.json")))
        merged.update(o["outcomes"])
        coll.update(o["collection_errors"])
        writes += o["worktree_writes_by_tests"]
    mc = {}
    for v in merged.values():
        mc[v] = mc.get(v, 0) + 1
    with open(os.path.join(d, "outcomes.json"), "w") as f:
        json.dump({"series": series, "arm": arm, "parts": n, "complete": prog["complete"], "counts": mc,
                   "n_ids": len(merged), "collection_errors": coll, "outcomes": merged,
                   "worktree_writes_by_tests": sorted(set(writes))}, f, indent=1)
    print("BWDF3 suite", series, arm, "part", n, "exit", rc, counts, "complete", prog["complete"],
          "collection errors", len(plugin.collection_errors), "new worktree writes", res["worktree_writes_by_tests"])
    return 0  # outcomes are compared per ID, not by exit code


if __name__ == "__main__":
    sys.exit(main())
