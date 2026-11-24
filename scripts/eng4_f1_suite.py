"""ENG4-F1 H3 suite runner (copy of scripts/eng4_suite.py with arms B and F1; F1 receipts under artifacts/eng4/f1/g4; restores F1). ENG4 G4 suite runner: production's test suite (tensor_cuda/tests + tests/) per test ID, for one build.

  python3 -B -P scripts/eng4_suite.py <series> <arm B|E> [--budget <seconds>]

- Installs the arm's engine files into the worktree first (the suite has source-reading tests, so each build runs
  with its own sources; ENG3's lesson), and its module into the worktree package:
    B: artifacts/eng4/before/* (git show HEAD:) + tensor_cuda/build-eng4/B/module/<so>
    E: artifacts/eng4/E_src/* (this round's final sources) + tensor_cuda/build-eng4/E/module/<so>
  Each module's sha256 must equal its build receipt (artifacts/eng4/build/BUILD_<arm>.json), else exit 3 (L103).
  After the run (and on any error) Build E's files and module are installed again and their sha256 re-checked.
- pytest options, identical for both builds: -q -p no:cacheprovider -rfE --rootdir=<root>
  --basetemp=/tmp/eng4_tmp/pt/<series>_<n> --junitxml -o junit_logging=all --continue-on-collection-errors;
  excluded: tensor_cuda/tests/test_eng4_matmul.py (this round's own tests, run as G1 / GPU lines) and
  tensor_cuda/tests/test_terrain_render.py (L40: it writes tracked artifacts/wo*.ppm files; excluded and listed).
- --budget: GPU-line chunking. The series keeps artifacts/eng4/g4/<series>/progress.json (IDs already run). A run
  deselects those IDs and stops after the first test that ends past the budget; the next line of the series
  continues. Without --budget the whole suite runs in one go (the CPU half).
- Writes artifacts/eng4/g4/<series>/part_<n>/outcomes.json and the merged artifacts/eng4/g4/<series>/outcomes.json:
  per test ID one of passed/failed/skipped/error/xfailed/xpassed, collection errors by file, and `git status
  --porcelain` (read-only git) before and after, so a write by a test into the worktree is listed.

Prior art: pytest's hook API (pytest_collection_modifyitems, pytest_runtest_logreport, pytest_collectreport); ENG3's
eng3_suite.py and eng3_srcswap.py (this repo), whose report plugin this copies. The resumable budgeted chunking is
mine; no prior art known to me beyond pytest's own --deselect / shouldstop.
"""
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
FILES = ["tensor_cuda/src/matmul.cu", "tensor_cuda/src/ops.cpp", "tensor_cuda/src/bindings.cpp",
         "tensor_cuda/include/tc/core.h", "tensor_cuda/tests/test_bwd_r1_matmul.py"]
SRC = {"B": "artifacts/eng4/before", "F1": "artifacts/eng4/f1/F1_src"}
RECEIPT = {"B": "artifacts/eng4/build/BUILD_B.json", "F1": "artifacts/eng4/f1/build/BUILD_F1.json"}
EXCLUDE = ["tensor_cuda/tests/test_eng4_matmul.py", "tensor_cuda/tests/test_terrain_render.py"]
SKIP_STATUS = ("artifacts/eng4/", "tensor_cuda/build-eng4/", "logs/eng4_", "scripts/eng4_",
               "tensor_cuda/tests/test_eng4_matmul.py") + tuple(FILES)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def install(arm):
    """Copy the arm's engine files and module into the worktree; return their sha256 after the copy."""
    rec = json.load(open(os.path.join(ROOT, RECEIPT[arm])))
    mod = os.path.join(ROOT, "tensor_cuda/build-eng4", arm, "module", SO)
    if sha(mod) != rec["module_sha256"]:
        raise SystemExit(f"REFUSED (L103): {mod} sha256 differs from BUILD_{arm}.json")
    got = {}
    for f in FILES:
        src = os.path.join(ROOT, SRC[arm], f.replace("/", "__"))
        dst = os.path.join(ROOT, f)
        shutil.copyfile(src, dst)
        got[f] = sha(dst)
        assert got[f] == sha(src), f
        if f in rec["sources"]:  # the engine sources must be the ones this module was built from
            assert got[f] == rec["sources"][f], (arm, f)
    tmp = MODULE + ".eng4tmp"
    shutil.copyfile(mod, tmp)
    os.replace(tmp, MODULE)  # atomic swap of the installed module (each arm keeps its receipted copy in build-eng4)
    got["module"] = sha(MODULE)
    assert got["module"] == rec["module_sha256"]
    return got


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
            self._session.shouldstop = f"eng4 budget {self.budget}s reached"


def main():
    series, arm = sys.argv[1], sys.argv[2]
    budget = int(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 0
    d = os.path.join(ROOT, "artifacts/eng4/f1/g4", series)
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
    try:
        installed = install(arm)
        os.environ["PYTHONPATH"] = PKG_PARENT
        os.environ.setdefault("TMPDIR", "/tmp/eng4_tmp")
        sys.path.insert(0, PKG_PARENT)
        import tensor_cuda as tc
        assert os.path.realpath(tc._C.__file__) == os.path.realpath(MODULE), tc._C.__file__
        import pytest
        args = ["-q", "-p", "no:cacheprovider", "-rfE", "--rootdir=" + ROOT,
                "--basetemp=/tmp/eng4_tmp/pt/" + series + f"_{n}", "--junitxml=" + os.path.join(part, "junit.xml"),
                "-o", "junit_logging=all", "--continue-on-collection-errors"]
        for e in EXCLUDE:
            args += ["--ignore=" + os.path.join(ROOT, e)]
        args += [os.path.join(ROOT, "tensor_cuda/tests"), os.path.join(ROOT, "tests")]
        plugin = Reports(set(prog["done"]), budget)
        os.chdir(ROOT)
        rc = int(pytest.main(args, plugins=[plugin]))
    finally:
        restored = install("F1")
    after = git_status()
    ran = dict(plugin.ids)
    counts = {}
    for v in ran.values():
        counts[v] = counts.get(v, 0) + 1
    res = {"series": series, "arm": arm, "part": n, "start": t0,
           "end": datetime.datetime.now(datetime.timezone.utc).isoformat(), "installed": installed,
           "restored_F1": restored, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
    print("ENG4 suite", series, arm, "part", n, "exit", rc, counts, "complete", prog["complete"],
          "collection errors", len(plugin.collection_errors), "new worktree writes", res["worktree_writes_by_tests"])
    return 0  # outcomes are compared per ID, not by exit code


if __name__ == "__main__":
    sys.exit(main())
