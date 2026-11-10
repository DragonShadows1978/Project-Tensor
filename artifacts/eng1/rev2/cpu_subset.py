#!/usr/bin/env python3
"""ENG1-REV2 Q4/Q6: run test_eng1.py's CPU subset (-k SELECTION) on one pinned arm, CUDA hidden.

Usage (cwd = worktree root, via run.py): python3 -B cpu_subset.py ARM SELECTION TAG
Mirrors the staged child's binding: the arm's package parent first on PYTHONPATH and sys.path (preload), the arm's
own test_eng1.py copy, pytest --import-mode=importlib, cacheprovider off; writes only under rev2/ and /tmp scratch.
Prior art: pytest -k selection and JUnit XML (pytest project); REV1/A1 arm-preload method. No new algorithm.
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path

REV2 = Path(__file__).resolve().parent
ROOT = REV2.parents[2]
TOOL = ROOT / "artifacts/eng1/tool"
arm, selection, tag = sys.argv[1:4]
package = TOOL / "arms" / arm / "tensor_cuda"
module = next((package / "tensor_cuda").glob("_tensor_cuda*.so"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
env = dict(os.environ)
env["PYTHONPATH"] = str(package.resolve())
assert env["CUDA_VISIBLE_DEVICES"] == ""
code = """import sys,json,pytest;from pathlib import Path
sys.path.insert(0,sys.argv[1]);import tensor_cuda as tc
class P:
 ids=[];reports=[]
 def pytest_collection_finish(self,session):self.ids=[i.nodeid for i in session.items]
 def pytest_runtest_logreport(self,report):self.reports.append(dict(id=report.nodeid,when=report.when,outcome=report.outcome,detail=str(report.longrepr)[-600:] if report.failed else ''))
p=P();rc=int(pytest.main(sys.argv[3:],plugins=[p]))
Path(sys.argv[2]).write_text(json.dumps(dict(exit=rc,package=tc.__file__,engine=tc._C.__file__,collected_ids=p.ids,reports=p.reports),indent=1))
raise SystemExit(rc)
"""
scratch = Path("/tmp/eng1_rev2_work") / ("cpu_" + tag)
scratch.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "-B", "-c", code, str(package), str(REV2 / (tag + "_pytest.json")), "-q", "-p",
       "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(ROOT), "--basetemp=" + str(scratch / "bt"),
       "--junitxml=" + str(REV2 / (tag + ".xml")), "-k", selection, str(package / "tests/test_eng1.py")]
p = subprocess.run(cmd, env=env, cwd=str(scratch), capture_output=True, text=True, timeout=500)
res = json.loads((REV2 / (tag + "_pytest.json")).read_text())
calls = [r for r in res["reports"] if r["when"] == "call"]
summary = dict(arm=arm, selection=selection, child_exit=p.returncode, engine=res["engine"],
               engine_sha256=sha(res["engine"]), arm_test_file_sha256=sha(package / "tests/test_eng1.py"),
               worktree_test_file_sha256=sha(ROOT / "tensor_cuda/tests/test_eng1.py"),
               collected=len(res["collected_ids"]),
               passed=sorted(r["id"] for r in calls if r["outcome"] == "passed"),
               failed=sorted(r["id"] for r in calls if r["outcome"] == "failed"),
               failed_detail={r["id"]: r["detail"][-200:] for r in calls if r["outcome"] == "failed"},
               non_call_failures=[r for r in res["reports"] if r["when"] != "call" and r["outcome"] != "passed"],
               stdout_tail=p.stdout[-800:])
(REV2 / (tag + "_summary.json")).write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in summary.items() if k != "stdout_tail"}))
print(p.stdout[-400:])
