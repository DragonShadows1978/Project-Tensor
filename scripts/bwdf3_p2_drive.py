"""BWD-F3 L48 CPU geometry drive of the P2 cases (no GPU; CUDA_VISIBLE_DEVICES empty): each P2 row of
scripts/bwdf3_p2_cases.py in its own child on a package, with CUDA tensors with data replaced by host tensors (the CPU
cannot hold them) and empty CUDA tensors kept. It records how far each row's host path goes (the refusal it reaches,
or the first device step), to catch case-geometry slips before the lead's GPU run; it is not a gate.

  python3 -B -P scripts/bwdf3_p2_drive.py <arm F3|F2|...>   -> artifacts/bwdf1/f3/gpu_dry/P2_GEOMETRY_DRIVE_<arm>.json
Prior art: none beyond the case table.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import numpy as np
import tensor_cuda as tc
import bwdf3_p2_cases as C
import bwdf1_gpu_common as G1
mode, rid = sys.argv[3], sys.argv[4]
def T(shape, dt="float32", salt=0):
    if dt == "uint8":
        return tc.tensor((np.arange(int(np.prod(shape))) % 251).astype(np.uint8).reshape(shape), device="cpu", dtype=dt)
    return tc.tensor(G1.data(shape, dt, salt), device="cpu", dtype=dt)
def E(shape, dt="float32"):
    return tc.tensor(np.zeros(shape, np.uint8 if dt == "uint8" else np.float32), device="cuda", dtype=dt)
row = {"id": rid}
with tc.no_grad():
    try:
        b, c = C.run_row(tc, mode, rid, T, E)
        o = b()
    except Exception as e:
        row["setup"] = f"{type(e).__name__}: {e}"[:300]
        print("ROW " + json.dumps(row)); raise SystemExit
    try:
        r = c(o)
        row["result"] = "RETURNED " + (json.dumps({k: list(v.shape) for k, v in r.items()}) if isinstance(r, dict) else "")
    except Exception as e:
        row["result"] = f"RAISED {type(e).__name__}: {e}"[:300]
print("ROW " + json.dumps(row))
'''


def main():
    arm = sys.argv[1]
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit("CPU only: CUDA_VISIBLE_DEVICES must be empty")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import bwdf3_p2_cases as C
    pkg = os.path.join(ROOT, "tensor_cuda/build-bwdf1", arm, "pkg")
    out = []
    for mode in C.MODES:
        for rid, kind in C.rows_for(mode):
            p = subprocess.run([sys.executable, "-B", "-P", "-c", CHILD, pkg, os.path.join(ROOT, "scripts"), mode, rid],
                               capture_output=True, text=True, timeout=120)
            got = [l[4:] for l in p.stdout.splitlines() if l.startswith("ROW ")]
            row = json.loads(got[-1]) if got else {"id": rid, "result": f"child exit {p.returncode}: {p.stderr[-300:]}"}
            row.update(mode=mode, kind=kind, child_exit=p.returncode)
            out.append(row)
            print(mode, rid, kind, row.get("setup") or row.get("result"))
    d = os.path.join(ROOT, "artifacts/bwdf1/f3/gpu_dry")
    os.makedirs(d, exist_ok=True)
    json.dump({"arm": arm, "rows": out}, open(os.path.join(d, f"P2_GEOMETRY_DRIVE_{arm}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
