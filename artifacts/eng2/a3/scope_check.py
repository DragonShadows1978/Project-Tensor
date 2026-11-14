#!/usr/bin/env python3
"""ENG2-A3 scope gate (seat form, no git; adapted from a2/scope_check.py): compare a3/worktree_before_a3.json (taken
before any A3 edit) with an after-map (artifacts/eng2/snapshot.py, unchanged).
Allowed by orders/ENG2_A3_ENG1_ROWS_AND_CONSUMERS.md:
  changed: tensor_cuda/tests/test_eng1.py (same line count; only lines 180, 186, 187, 266 differ: the three dtype lists
           and one comment), artifacts/eng2/L38_PINS.json (expected_changes emptied, one "a3" record added, every other
           key equal), and the staging files that re-collecting and re-sealing rewrite (tool/staging_seal.json,
           tool/arms/**, tool/ids_*.txt, tool/collect_*.json, tool/baseline_expected_failures.json, tool/gpu_lines.json,
           tool/regression_files.json, tool/source_reading_tests.json, tool/STAGED_GPU_COMMANDS.sh);
  added:   artifacts/eng2/a3/**, and the fresh dry-run and audit roots tool/gpu_runs/dry_4/**, tool/gpu_runs/audit_4/**
           (the order names dry_3/audit_3 "(new)", which A2 had already written: deviation D1);
  removed: nothing.
Usage: scope_check.py AFTER.json   -> a3/A3_SCOPE.json
Prior art: manifest diffing (BSD mtree(8); unverified -- lead to check); difflib-free line comparison. No new algorithm.
"""
import json
from pathlib import Path
import sys

A3 = Path(__file__).resolve().parent
ROOT = A3.parents[2]
before = json.loads((A3 / "worktree_before_a3.json").read_text())["files"]
after = json.loads(Path(sys.argv[1]).read_text())["files"]
T = "artifacts/eng2/tool/"
STAGING = {T + n for n in ("staging_seal.json", "baseline_expected_failures.json", "gpu_lines.json", "regression_files.json",
                           "source_reading_tests.json", "STAGED_GPU_COMMANDS.sh")}
staging_ok = lambda k: k in STAGING or k.startswith((T + "arms/", T + "ids_", T + "collect_"))
CODE = {"tensor_cuda/tests/test_eng1.py", "artifacts/eng2/L38_PINS.json"}
ADDED_OK = ("artifacts/eng2/a3/", T + "gpu_runs/dry_4/", T + "gpu_runs/audit_4/")
changed = sorted(k for k in before if k in after and before[k] != after[k])
removed = sorted(k for k in before if k not in after)
added = sorted(k for k in after if k not in before)
bad_changed = [k for k in changed if k not in CODE and not staging_ok(k) and not k.startswith("artifacts/eng2/a3/")]
bad_added = [k for k in added if not k.startswith(ADDED_OK)]
old = (A3 / "before/tensor_cuda__tests__test_eng1.py").read_text().splitlines()
new = (ROOT / "tensor_cuda/tests/test_eng1.py").read_text().splitlines()
lines = [i + 1 for i, (a, b) in enumerate(zip(old, new)) if a != b] if len(old) == len(new) else None
t1 = dict(lines_before=len(old), lines_after=len(new), differing_lines=lines,
          ok=lines == [180, 186, 187, 266] and before["tensor_cuda/tests/test_eng1.py"] != after["tensor_cuda/tests/test_eng1.py"])
pb, pa = json.loads((A3 / "before/artifacts__eng2__L38_PINS.json").read_text()), json.loads((ROOT / "artifacts/eng2/L38_PINS.json").read_text())
others_equal = {k: v for k, v in pb.items() if k != "expected_changes"} == {k: v for k, v in pa.items() if k not in ("expected_changes", "a3")}
pins = dict(expected_changes_before=len(pb["expected_changes"]), expected_changes_after=pa["expected_changes"],
            other_keys_equal=others_equal, added_keys=sorted(set(pa) - set(pb)),
            ok=pa["expected_changes"] == {} and others_equal and sorted(set(pa) - set(pb)) == ["a3"])
arm_changes = {}
for k in changed:
    if k.startswith(T + "arms/"):
        arm, rest = k[len(T + "arms/"):].split("/", 1)
        arm_changes.setdefault(rest, []).append(arm)
arms_ok = list(arm_changes) == ["tensor_cuda/tests/test_eng1.py"] and len(arm_changes["tensor_cuda/tests/test_eng1.py"]) == 17
staging_changed = sorted(k for k in changed if staging_ok(k) and not k.startswith(T + "arms/"))
ok = not bad_changed and not bad_added and not removed and t1["ok"] and pins["ok"] and arms_ok
rec = dict(before_files=len(before), after_files=len(after),
           changed_outside_a3_and_arms=[k for k in changed if not k.startswith("artifacts/eng2/a3/") and not k.startswith(T + "arms/")],
           arm_file_changes={k: sorted(v) for k, v in arm_changes.items()}, arms_ok=arms_ok, staging_changed=staging_changed,
           test_eng1=t1, l38_pins=pins, removed=removed,
           added_by_area={p: sum(k.startswith(p) for k in added) for p in ADDED_OK},
           violations=dict(changed=bad_changed, added=bad_added, removed=removed), passed=ok)
(A3 / "A3_SCOPE.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec, indent=1))
sys.exit(0 if ok else 1)
