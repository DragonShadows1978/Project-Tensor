#!/usr/bin/env python3
"""ENG2 G4: the L38 branch of tool/evidence.py regression_reasons on constructed child results (CPU, no tool writes).
Cases: (1) every pin changes as registered and nothing else is lost -> no reasons; (2) a pin keeps passing -> reason;
(3) a pin fails with another text -> reason; (4) a non-pinned baseline-passing ID fails -> reason; (5) suite without
pins (regdtype-like) -> no reasons. Prior art: none known to me (constructed-receipt unit check, as ENG1-A2's audit
shape classes).
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import evidence  # noqa: E402

pins = json.loads((Path(__file__).resolve().parents[1] / "L38_PINS.json").read_text())["expected_changes"]
P = "artifacts/eng2/tool/arms/candidate/"  # arm prefix is stripped by norm_id only for arm paths; worktree IDs are plain
other = "tensor_cuda/tests/test_eng1.py::test_e2_pow_grad[float32-p+2]"
ids = sorted(pins) + [other]
text = "E   TypeError: unsupported dtype 'float64'; supported dtypes: float32, float16, int64, bool, uint8, bfloat16"


def child(outcomes, details=None):
    reports = [dict(id=i, when="call", outcome=o, detail=(details or {}).get(i, text if o == "failed" else ""))
               for i, o in outcomes.items()]
    return dict(collected_ids=list(outcomes), reports=reports)


base = child({i: "passed" for i in ids})
cases = {
    "1_as_registered": child({**{i: "failed" for i in pins}, other: "passed"}),
    "2_pin_still_passes": child({**{i: "failed" for i in pins}, sorted(pins)[0]: "passed", other: "passed"}),
    "3_pin_other_text": child({**{i: "failed" for i in pins}, other: "passed"}, {sorted(pins)[1]: "E   RuntimeError: x"}),
    "4_unpinned_lost": child({**{i: "failed" for i in pins}, other: "failed"}),
}
results = {k: evidence.regression_reasons(base, v) for k, v in cases.items()}
results["5_no_pins_in_suite"] = evidence.regression_reasons(child({other: "passed"}), child({other: "passed"}))
ok = (results["1_as_registered"] == [] and len(results["2_pin_still_passes"]) == 1
      and len(results["3_pin_other_text"]) == 1 and len(results["4_unpinned_lost"]) == 1
      and results["5_no_pins_in_suite"] == [])
print(json.dumps(dict(results=results, ok=ok), indent=1))
assert ok
