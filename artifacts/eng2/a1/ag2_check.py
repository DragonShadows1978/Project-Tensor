#!/usr/bin/env python3
"""ENG2-A1 A-G2 scoring of the CPU receipts (a1/<arm>_cpu.json from a1/cpu_runs.py, the unchanged copy of g2/cpu_runs.py).
  G2a candidate: every CPU ID of test_eng2.py passes (157 = 121 earlier + 36 new).
  G2b baseline: the failing CPU IDs == the 60 registered F1 CPU failures (artifacts/eng2/g2/baseline_cpu.json) + the
      36 new non-string IDs, exactly; the regenerated tool/baseline_expected_failures.json == the earlier 106 + those 36.
  G2c MF1 / MF2: failing IDs == the registered killers, each killer traced through the registered hunk (L41).
  G2d test_eng2.py additions only: the pre-A1 file is a byte prefix of the new one; the census gains exactly the 36
      test_a1_* IDs and loses none.
Prior art: set-equality scoring of per-ID outcomes (GN1/ENG1 tooling); no new algorithm.
"""
import json
from pathlib import Path
import sys

A1 = Path(__file__).resolve().parent
ENG2 = A1.parent
ROOT = ENG2.parents[1]
r = lambda p: json.loads(Path(p).read_text())
out = {}
cand, base = r(A1 / "candidate_cpu.json"), r(A1 / "baseline_cpu.json")
new_ids = sorted(set((ENG2 / "tool/ids_eng2.txt").read_text().split()) - set((A1 / "before/artifacts__eng2__tool__ids_eng2.txt").read_text().split()))
lost = sorted(set((A1 / "before/artifacts__eng2__tool__ids_eng2.txt").read_text().split()) - set((ENG2 / "tool/ids_eng2.txt").read_text().split()))
a1_ids = sorted(i for i in (ENG2 / "tool/ids_eng2.txt").read_text().split() if "::test_a1_" in i)
out["G2a"] = dict(collected=cand["n"], failed=sorted(k for k, v in cand["outcomes"].items() if v != "passed"),
                  ok=cand["n"] == 157 and cand["rc"] == 0 and all(v == "passed" for v in cand["outcomes"].values())
                  and len(cand["outcomes"]) == 157)
old_f1 = sorted(k for k, v in r(ENG2 / "g2/baseline_cpu.json")["outcomes"].items() if v != "passed")
base_failed = sorted(k for k, v in base["outcomes"].items() if v != "passed")
reg_new = r(ENG2 / "tool/baseline_expected_failures.json")
reg_old = r(A1 / "before/artifacts__eng2__tool__baseline_expected_failures.json")
out["G2b"] = dict(collected=base["n"], failed=len(base_failed), registered_f1_cpu=len(old_f1), new_nonstring=len(a1_ids),
                  equal=base_failed == sorted(old_f1 + a1_ids),
                  registration=dict(before=len(reg_old), after=len(reg_new), after_equals_before_plus_new=sorted(reg_new) == sorted(reg_old + a1_ids)),
                  ok=base_failed == sorted(old_f1 + a1_ids) and len(old_f1) == 60 and len(a1_ids) == 36
                  and sorted(reg_new) == sorted(reg_old + a1_ids))
mutants = {m["id"]: m for m in r(ENG2 / "mutants/mutants.json")}
for arm in ("MF1", "MF2"):
    d = r(A1 / f"{arm}_cpu.json")
    failed = sorted(k for k, v in d["outcomes"].items() if v != "passed")
    killers = sorted(mutants[arm]["killers"])
    out["G2c_" + arm] = dict(failed=len(failed), killers=len(killers), equal=failed == killers,
                             traced=sum(bool(d["mutant_trace"].get(k)) for k in killers), hunk_lines=mutants[arm]["hunk_lines"],
                             ok=failed == killers and all(d["mutant_trace"].get(k) for k in killers))
before_t = (A1 / "before/tensor_cuda__tests__test_eng2.py").read_bytes()
now_t = (ROOT / "tensor_cuda/tests/test_eng2.py").read_bytes()
out["G2d"] = dict(prefix=now_t.startswith(before_t), bytes_before=len(before_t), bytes_after=len(now_t),
                  ids_added=len(new_ids), ids_lost=lost, added_are_a1=new_ids == a1_ids,
                  ok=now_t.startswith(before_t) and not lost and new_ids == a1_ids and len(new_ids) == 36)
out["new_ids"] = new_ids
out["all_ok"] = all(v["ok"] for k, v in out.items() if isinstance(v, dict) and "ok" in v)
(A1 / "AG2_CPU.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: {q: w for q, w in v.items() if q not in ("failed",) or isinstance(w, int)} for k, v in out.items() if isinstance(v, dict)}, indent=1))
print("all_ok", out["all_ok"])
sys.exit(0 if out["all_ok"] else 1)
