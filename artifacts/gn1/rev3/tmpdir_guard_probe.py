#!/usr/bin/env python3
"""GN1-REV3 Q6: A3's REAL scratch_guard (mirror copy, sealed bytes) on hostile
TMPDIR strings, including REV2's R2-F10 escape. Pure path logic; no writes.
Prior art: POSIX realpath(3) canonicalization; no other prior art known to me."""
import json, os, sys
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
sys.path.insert(0, "/tmp/gn1_rev3_m/W/artifacts/gn1/a3")
from evidence import scratch_guard  # noqa: E402
cases = {"/tmp/gn1_a3_boot/../gn1_rev2_tmp": False, "/tmp/gn1_a3_x/../../etc": False, "/tmp/gn1_a3_x/sub": False,
         "gn1_a3_rel": False, "/tmp//gn1_a3_ok": True, "/tmp/./gn1_a3_ok": True, "/tmp/gn1_a3_ok/.": True,
         "/tmp/gn1_a3_staged": True, "/var/tmp/gn1_a3_x": False, "/tmp/gn1_a3_x/../gn1_a3_y": True}
rows = {}
for value, expect in cases.items():
    try:
        out = str(scratch_guard(value)); ok = True
    except AssertionError as exc:
        out = str(exc); ok = False
    rows[value] = dict(accepted=ok, resolved_or_error=out, expected_accept=expect, as_expected=ok == expect)
res = dict(all_as_expected=all(r["as_expected"] for r in rows.values()), cases=rows)
(Path(__file__).resolve().parent / "tmpdir_guard_probe.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=1))
