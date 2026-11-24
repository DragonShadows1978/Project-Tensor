import subprocess
if subprocess.run(["forge-gpu-lease-check"]).returncode != 0: raise SystemExit("not inside a forge-gpu-run lease")
# BWD-F1 G2 direct witness (staged GPU line, run under compute-sanitizer memcheck by the lead): every binary op in
# artifacts/bwdf1/CENSUS.json on [4, 0] against [4, 1], each dtype, forward + backward, then one nonzero-shape
# control per op saved for the bitwise B/E comparison, then the H3 current-device message (native self-test).
# Usage: python3 -B -P scripts/bwdf1_step_g2_direct.py <arm B|E>; writes only artifacts/bwdf1/gpu/g2_direct_<arm>/
# (refuses to start if it exists). Prior art: see scripts/bwdf1_gpu_common.py.
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bwdf1_gpu_common as G  # noqa: E402

arm = sys.argv[1]
OUT = G.GPU / f"g2_direct_{arm}"
if OUT.exists():
    raise SystemExit(f"{OUT} exists; move it aside before a retry")
OUT.mkdir(parents=True)
tc, sha = G.load_arm(arm)
import numpy as np  # noqa: E402

rows, arrays = [], {}
for c in G.cases() + G.controls():
    row, arr = G.run_case(tc, c)
    if not c.get("control"):
        row["verdict"] = G.case_verdict(row)
    rows.append(row)
    for k, v in arr.items():
        if v is not None:
            arrays[f"{c['id']}|{k}"] = v
    print(json.dumps(row), flush=True)
try:
    st = tc._C._bwd_native_selftest("current-device")
except Exception as e:  # noqa: BLE001
    st = {"error": f"{type(e).__name__}: {e}"}
np.savez(OUT / "controls.npz", **{k: v for k, v in arrays.items() if k.startswith("control-")})
zero = [r for r in rows if "verdict" in r]
verdict = "PASS" if all(r["verdict"] == "PASS" for r in zero) else "FAIL"
(OUT / "results.json").write_text(json.dumps({"line": f"g2_direct_{arm}", "arm": arm, "module_sha256": sha,
                                              "rows": rows, "h3_current_device_selftest": st,
                                              "zero_case_verdict": verdict}, indent=1, default=str))
print("H3 current-device:", json.dumps(st, default=str))
print("VERDICT", verdict, sum(r["verdict"] == "PASS" for r in zero), "/", len(zero))
