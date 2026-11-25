"""ENG5-S-T replay drive (a CPU preview of T2, not a GPU run): the GPU-lane test function of
tensor_cuda/tests/test_eng5s_status_checks.py is called on the rows ENG5-S's lead recorded on the GPU
(artifacts/eng5s/gpu/g3_<arm>/g3_rows.json: the same child bodies, copied), with run_row replaced by a lookup. Each
row's test outcome becomes a junit file (artifacts/eng5s/t/t3_replay_junit/<arm>.xml) that scripts/eng5st_verdict.py
judges. It shows what the test's assertions make of real GPU observations; it does not replace the lead's T2 lines
(the test's children are not run here). Site 9's recorded row (packed_prefill_dump) is not a row of this test.

  CUDA_VISIBLE_DEVICES= python3 -B -P scripts/eng5st_replay.py     -> artifacts/eng5s/t/T3B_REPLAY.json

Prior art: no prior art known to me (replaying recorded observations through a test's assertions is ordinary
golden-file testing).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tensor_cuda/tests"))
sys.path.insert(0, os.path.join(ROOT, "tensor_cuda"))   # the installed package, for the test module's import only
import eng5st_verdict as vd
import test_eng5s_status_checks as T

ARMS = {"S": "g3_S", "B": "g3_B", "s01": "g3_s01", "s02": "g3_s02", "s03": "g3_s03", "s04": "g3_s04"}


def main():
    out = {"source": "artifacts/eng5s/gpu/g3_<arm>/g3_rows.json (ENG5-S lead's GPU lines, recorded)", "arms": {}}
    jd = os.path.join(ROOT, "artifacts/eng5s/t/t3_replay_junit")
    os.makedirs(jd, exist_ok=True)
    bad = []
    want = {"S": "PASS", "B": "DEFECT", "s01": "KILLED", "s02": "KILLED", "s03": "KILLED", "s04": "KILLED"}
    for arm, line in ARMS.items():
        rows = {r["case"]: r for r in json.load(open(os.path.join(ROOT, "artifacts/eng5s/gpu", line, "g3_rows.json")))}
        T.run_row = lambda inject_so, case, rows=rows: rows[case]
        status, why = {}, {}
        for row in T.GPU_ROWS + T.GPU_CONTROL_ONLY:
            try:
                T.test_gpu_status_rows(row, None)
                status[row] = "passed"
            except AssertionError as ex:
                status[row] = "killed" if vd.SIGNATURE in str(ex) else "failed_other"
                why[row] = str(ex).splitlines()[0][:200]
        p = os.path.join(jd, f"{arm}.xml")
        open(p, "w").write(vd.junit_xml(status))
        marks = [MARK for MARK in [vd.MUTANT_IDS.get(arm)] if MARK] if arm in vd.MUTANT_IDS else None
        v = vd.judge(arm, p, marks)
        ok = v["verdict"] == want[arm]
        out["arms"][arm] = {"junit": os.path.relpath(p, ROOT), "statuses": status, "failure_heads": why,
                            "verdict": v["verdict"], "expected": want[arm], "ok": ok,
                            "marker_note": "marker lines are assumed present here (the replay runs no mutant)"}
        if not ok:
            bad.append(arm)
        print(arm, v["verdict"], "expected", want[arm], "OK" if ok else "MISMATCH")
    out["mismatches"] = bad
    json.dump(out, open(os.path.join(ROOT, "artifacts/eng5s/t/T3B_REPLAY.json"), "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
