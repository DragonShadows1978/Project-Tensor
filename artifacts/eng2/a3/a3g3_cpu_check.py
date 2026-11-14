#!/usr/bin/env python3
"""ENG2-A3 A3-G1/A3-G3 scoring of this round's CPU receipts and re-collected censuses (adapted from a2/ag2_check.py;
the registered numbers are a3/LEDGER.md RG1, RG2, RG4, RG5). Comparisons are by normalized test ID and outcome,
exactly (L09); logs and timing are excluded.
  C1 (RG4) candidate test_eng2.py `-k cpu`: 253 collected, all passed, pytest rc 0.
  C2 (RG4) baseline test_eng2.py `-k cpu`: the failing set == A2's (a2/baseline_cpu.json) == A1's 96 registered F1 CPU
     IDs + A2's 24 non-string sentinel IDs, exactly; everything else passes.
  C3 (RG4) test_eng1.py `-k cpu` on both arms: 47 collected, all passed; the 47 IDs == the pre-A3 CPU subset (no float64
     ID was a CPU ID), so the edit changed no CPU ID.
  C4 MF1 / MF2 (A2's method): failing IDs == the registered killers, each traced through the registered hunk.
  C5 (RG1) test_eng1.py IDs 156 -> 146 on both arms, removed == the 10 pre-A3 L38 pins, none added.
  C6 (RG2) re-collected censuses: ids_regnorm(_candidate) == before minus exactly the 10; ids_regdtype(_candidate),
     ids_eng2(_candidate), ids_audit byte-identical; each collect_*.json rc 0, no errors.
  C7 (RG5) gpu_lines.json, baseline_expected_failures.json, STAGED_GPU_COMMANDS.sh, regression_files.json,
     source_reading_tests.json byte-identical to before/; L38_PINS.json expected_changes == {}.
Prior art: set-equality scoring of per-ID outcomes (GN1/ENG1 tooling); no new algorithm.
"""
import json
from pathlib import Path
import sys

A3 = Path(__file__).resolve().parent
ENG2 = A3.parent
TOOL = ENG2 / "tool"
B = lambda n: A3 / "before" / ("artifacts__eng2__tool__" + n)
r = lambda p: json.loads(Path(p).read_text())
failing = lambda d: sorted(k for k, v in d["outcomes"].items() if v != "passed")
out = {}
cand, base = r(A3 / "candidate_cpu.json"), r(A3 / "baseline_cpu.json")
out["C1"] = dict(collected=cand["n"], failed=failing(cand),
                 ok=cand["n"] == 253 and cand["rc"] == 0 and len(cand["outcomes"]) == 253 and not failing(cand))
a2_fail, a1_fail = failing(r(ENG2 / "a2/baseline_cpu.json")), failing(r(ENG2 / "a1/baseline_cpu.json"))
sentinel = sorted(i for i in r(ENG2 / "a2/baseline_cpu.json")["outcomes"] if "::test_a2_shape_sentinel_refused_cpu[" in i)
out["C2"] = dict(collected=base["n"], failed=len(failing(base)), equals_a2=failing(base) == a2_fail,
                 equals_a1_f1_plus_sentinel=failing(base) == sorted(a1_fail + sentinel), a1_f1=len(a1_fail),
                 sentinel=len(sentinel),
                 ok=failing(base) == a2_fail == sorted(a1_fail + sentinel) and len(a1_fail) == 96 and len(sentinel) == 24
                 and base["n"] == 253)
pre_cpu = sorted(i for i in (A3 / "ids_test_eng1_before_candidate.txt").read_text().split() if "cpu" in i.lower())
for arm in ("baseline", "candidate"):
    d = r(A3 / f"{arm}_eng1_cpu.json")
    out["C3_" + arm] = dict(collected=d["n"], failed=failing(d), ids_equal_pre_a3_cpu_subset=sorted(d["outcomes"]) == pre_cpu,
                            ok=d["rc"] == 0 and d["n"] == 47 and not failing(d) and sorted(d["outcomes"]) == pre_cpu)
mutants = {m["id"]: m for m in r(ENG2 / "mutants/mutants.json")}
for arm in ("MF1", "MF2"):
    d = r(A3 / f"{arm}_cpu.json")
    killers = sorted(mutants[arm]["killers"])
    out["C4_" + arm] = dict(failed=len(failing(d)), killers=len(killers), equal=failing(d) == killers,
                            traced=sum(bool(d["mutant_trace"].get(k)) for k in killers), hunk_lines=mutants[arm]["hunk_lines"],
                            ok=failing(d) == killers and all(d["mutant_trace"].get(k) for k in killers))
ids = {k: (A3 / f"ids_test_eng1_{k}.txt").read_text().split() for k in
       ("before_candidate", "before_baseline", "after_candidate", "after_baseline")}
pins = sorted(r(A3 / "before/artifacts__eng2__L38_PINS.json")["expected_changes"])
removed = sorted(set(ids["before_candidate"]) - set(ids["after_candidate"]))
out["C5"] = dict(before=len(ids["before_candidate"]), after=len(ids["after_candidate"]), removed=removed,
                 added=sorted(set(ids["after_candidate"]) - set(ids["before_candidate"])),
                 ok=removed == pins and len(pins) == 10 and ids["before_candidate"] == ids["before_baseline"]
                 and ids["after_candidate"] == ids["after_baseline"] and len(ids["after_candidate"]) == 146
                 and not set(ids["after_candidate"]) - set(ids["before_candidate"]))
c6 = {}
for n in ("regnorm", "regnorm_candidate"):
    old, new = B(f"ids_{n}.txt").read_text().split(), (TOOL / f"ids_{n}.txt").read_text().split()
    c6[n] = dict(before=len(old), after=len(new), lost_equals_pins=sorted(set(old) - set(new)) == pins,
                 gained=sorted(set(new) - set(old)), ok=sorted(set(old) - set(new)) == pins and set(new) <= set(old))
for n in ("regdtype", "regdtype_candidate", "eng2", "eng2_candidate", "audit"):
    c6[n] = dict(identical=B(f"ids_{n}.txt").read_bytes() == (TOOL / f"ids_{n}.txt").read_bytes())
    c6[n]["ok"] = c6[n]["identical"]
collects = sorted(TOOL.glob("collect_*.json"))
c6["collect_files"] = {p.name: [r(p)["rc"], r(p)["n"], len(r(p)["errors"])] for p in collects}
c6["collect_ok"] = all(v[0] == 0 and v[2] == 0 for v in c6["collect_files"].values()) and len(collects) == 7
out["C6"] = dict(c6, ok=all(v["ok"] for k, v in c6.items() if isinstance(v, dict) and "ok" in v) and c6["collect_ok"])
same = {n: B(n).read_bytes() == (TOOL / n).read_bytes() for n in
        ("gpu_lines.json", "baseline_expected_failures.json", "STAGED_GPU_COMMANDS.sh", "regression_files.json",
         "source_reading_tests.json")}
pins_now = r(ENG2 / "L38_PINS.json")
out["C7"] = dict(identical=same, expected_failures=len(r(TOOL / "baseline_expected_failures.json")),
                 lines=len(r(TOOL / "gpu_lines.json")), l38_expected_changes=pins_now["expected_changes"],
                 ok=all(same.values()) and pins_now["expected_changes"] == {})
out["all_ok"] = all(v["ok"] for k, v in out.items() if isinstance(v, dict) and "ok" in v)
(A3 / "A3_CPU_AND_CENSUS.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: {q: w for q, w in v.items() if not isinstance(w, list) or len(w) <= 10} for k, v in out.items()
                  if isinstance(v, dict)}, indent=1))
print("all_ok", out["all_ok"])
sys.exit(0 if out["all_ok"] else 1)
