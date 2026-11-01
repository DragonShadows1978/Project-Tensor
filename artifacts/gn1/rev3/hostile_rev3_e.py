#!/usr/bin/env python3
"""GN1-REV3 part E (Q2, Q3): A3's own real-test D2 controls re-run independently
through A3's REAL parent (parent_wrap.py) and scorer in the REV3 mirror, using
A3's hostile_forge.py real_* modes (real sealed test file + real Reports plugin
on the CPU stand-in engine); plus E5, a truncated child_result.json left by an
interrupted attempt. Reuses hostile_rev3.py helpers.
Prior art: as hostile_rev3.py. Ours: E5.
"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hostile_rev3 as H  # noqa: E402

def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    H.BASE.mkdir(parents=True)
    g = [H.PY, "hostile_forge.py", "--max", "30", "--"]
    tail = lambda arm, suite: ["timeout", "20s", "python3", "run_gpu_tests.py", arm, suite]
    for name, mode, accept, needle in (("E1_real_engine_generic_kill", "real_generic", True, None),
                                        ("E2_real_engine_device_veto", "real_device", False, "device-pattern failure"),
                                        ("E3_real_callee_origin", "real_callee", False, "no real non-reference")):
        runs = H.start(name)
        c = H.parent(runs, "candidate_focus", g, tail("candidate", "focus"), bound=60, HOSTILE_MODE="real_honest")
        r = H.parent(runs, "M07fast", g, tail("M07fast", "mutation"), bound=60, HOSTILE_MODE=mode)
        sc = H.score(runs, "M07fast")
        line = sc.get("line") or {}
        ok = (c["exit"] == 0 and (r["exit"] == 0 and line.get("passed") is True if accept else
              r["exit"] == 1 and any(needle in x for x in (r["parent"] or {}).get("reasons", [])) and not line.get("passed")))
        ev = sorted((runs / "M07fast").glob("attempt-*/real_test_evidence.json"))
        H.record(name, "R2-F02/F03 (A3 control, re-run)", "kill accepted" if accept else "rejected by: " + needle, ok,
                 candidate=c, run=r, scorer=sc, real_test_evidence=json.loads(ev[-1].read_text())["source"] if ev else None)
    runs = H.start("E4_real_candidate_not_passed")
    c = H.parent(runs, "candidate_focus", g, tail("candidate", "focus"), bound=60, HOSTILE_MODE="real_candidate_fail")
    r = H.parent(runs, "M07fast", g, tail("M07fast", "mutation"), bound=60, HOSTILE_MODE="real_generic")
    sc = H.score(runs)
    H.record("E4_real_candidate_not_passed", "R2-F02 (A3 control, re-run)", "mutation refused without passing candidate focus",
             c["exit"] == 1 and r["parent"] is None and "mutations require recomputed passing candidate focus" in r["stderr"] and sc["exit"] == 1,
             candidate=c, run=r, scorer=dict(exit=sc["exit"]))
    # E5: an interrupted attempt whose child_result.json is truncated (child killed mid-write, parent gone).
    runs = H.start("E5_truncated_child_result")
    att = runs / "candidate_focus" / "attempt-0001"
    att.mkdir(parents=True)
    (att / "child_result.json").write_text('{"arm": "candidate", "suite": "focus", "pytest_exit": 1, "reports": [')
    retry = H.parent(runs, "candidate_focus", [H.PY, str(H.OUT / "rev3_forge.py"), "--"],
                     ["timeout", "10s", "python3", "run_gpu_tests.py", "candidate", "focus"], REV2_MODE="normal")
    sc = H.score(runs, "candidate_focus")
    H.record("E5_truncated_child_result", "R2-F07 class (new)", "fail closed with a clean refusal (no traceback crash)",
             retry["exit"] != 0 and "Traceback" not in retry["stderr"] and not (sc.get("line") or {}).get("passed"),
             retry=retry, scorer=sc)
    (H.OUT / "hostile_rev3_e.json").write_text(json.dumps(dict(base=str(H.BASE), results=H.RESULTS), indent=2) + "\n")
    print(json.dumps([(x["case"], x["outcome_as_expected"]) for x in H.RESULTS]))

if __name__ == "__main__":
    main()
