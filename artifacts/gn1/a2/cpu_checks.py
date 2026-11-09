"""G2/G5 real Python tests with GPU hidden; NumPy stand-in is explicit.
Prior art: pytest (Krekel et al.); sys.settrace/REV1; mutation RIP model
(Offutt/Voas, unverified — lead to check). A2 records real edit-site reach,
loading and assertion failures on CPU, never GPU mutant kills.
"""
import ast
import contextlib
import difflib
import json
import os
from pathlib import Path
import sys
import pytest
from cpu_shim import install
from run_gpu_tests import Reports

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[2]
TEST = ROOT / "tensor_cuda/tests/test_groupnorm_gn1.py"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ["TMPDIR"].startswith("/tmp/gn1_a2_")


def run(path, ids, logfile, trace=False):
    executed = set()
    def tracer(frame, event, arg):
        if frame.f_code.co_filename == str(path):
            if event == "line":
                executed.add(frame.f_lineno)
            return tracer
        return None
    if trace:
        sys.settrace(tracer)
    plugin = Reports()
    try:
        install(path)
        with logfile.open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            args = ["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir="+str(ROOT), *ids]
            rc = int(pytest.main(args, plugins=[plugin]))
    finally:
        sys.settrace(None)
    calls = [r for r in plugin.reports if r["when"] == "call"]
    return dict(exit=rc, collected_ids=plugin.ids, reports=plugin.reports,
                collection_errors=plugin.collection_errors, executed_lines=sorted(executed),
                failed_ids=[r["id"] for r in calls if r["outcome"] == "failed"])


candidate = TOOL / "arms/candidate/tensor_cuda/nn.py"
mode = sys.argv[1]
if mode in ("candidate", "baseline"):
    candidate = TOOL / "arms" / mode / "tensor_cuda/nn.py"
    result = run(candidate, [str(TEST)], TOOL / ("cpu_"+mode+"_shim.log"))
    result["evidence_class"] = "CPU NumPy stand-in execution of real Python class/test suite; not engine/GPU"
    (TOOL / ("G2_shim.json" if mode=="candidate" else "baseline_shim.json")).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(exit=result["exit"], ids=len(result["collected_ids"]), failures=result["failed_ids"])))
    sys.exit(result["exit"])
assert mode == "mutants"
rows = []
for m in json.loads((TOOL / "mutants.json").read_text()):
    mid = m["id"]
    path = TOOL / "mutants" / mid / "nn.py"
    assert ast.parse(path.read_text()) and path.read_bytes() != candidate.read_bytes()
    result = run(path, [m["predicted_witness"]], TOOL / "mutants" / mid / "cpu_witness.log", True)
    changed = sorted({j+1 for tag,a,b,c,d in difflib.SequenceMatcher(None, candidate.read_text().splitlines(), path.read_text().splitlines()).get_opcodes() if tag!="equal" for j in range(c,d)})
    hit = sorted(set(changed) & set(result["executed_lines"]))
    calls = [r for r in result["reports"] if r["when"] == "call"]
    killed = any(r["outcome"] == "failed" and r.get("exception_class") == "AssertionError" for r in calls)
    equivalent = bool(m["equivalence"]) and result["exit"] == 0
    row = dict(id=mid, witness=m["predicted_witness"], edited_lines=changed, reached_lines=hit,
               reached=bool(hit), loads=True, differs=True, cpu_assertion_killed=killed,
               equivalence=m["equivalence"], passed=bool(hit) and not result["collection_errors"] and (killed or equivalent), trace=result)
    (TOOL / "mutants" / mid / "trace.json").write_text(json.dumps(row, indent=2)+"\n")
    rows.append(row)
    print(mid, "PASS" if row["passed"] else "FAIL", "edited", changed, "reached", hit, "failed", result["failed_ids"])
result = dict(passed=all(r["passed"] for r in rows), mutants=rows, gpu_kills_claimed=False)
(TOOL / "G5_mutants.json").write_text(json.dumps(result, indent=2)+"\n")
sys.exit(0 if result["passed"] else 1)
