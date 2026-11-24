import subprocess
if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")
# BWD-F3 P2 (staged GPU line): the E1-E4 cases that only the GPU can show, one mode per line, each row in its own child
# process (so a host or device fault on Build B / Build F2 / a mutant is recorded per row and cannot hide the others;
# L48). Usage: python3 -B -P scripts/bwdf3_step_p2.py <mode> <arm>; writes only artifacts/bwdf1/f3/gpu/p2_<mode>_<arm>/
# (refuses to start if it exists). Modes (operands are CUDA tensors with data unless named empty):
#   e1    (memcheck) RoPE read spans on reading launches: export_rope_rows / export_row_pairs / swap / arena with a
#         short sin, a negative pos0 or a short cos are refused before any work; the empty launches with CUDA tables
#         return their shapes; controls (covering tables) keep their values (B vs F3 bitwise).
#   e2    (memcheck) table layout: [T, D, 1] / [T, D, 1, 1] tables give results bitwise equal to the [T, D] tables in
#         the same build (rope_apply, export_rope_rows, swap); a [T, D, 2] table is refused.
#   e3    (memcheck) reduction axes: sum([0, 0]) on [2] (the review's case: the tail kernel reads 4 elements of a
#         2-element buffer on Build F2), sum([8]) on rank 1, mean([0, 0]), softmax(-3), argmax(2) refused before any
#         work; valid-axis controls keep their values (B vs F3 bitwise).
#   count (no memcheck; ENG3's LD_PRELOAD counter in each row's child) E4 ordering: a first pair that allocates and
#         launches, then a later pair whose output's bytes exceed int64 ([0, 2^62, 2] uint8): refused with no
#         allocation, launch or copy during the call (Build F2 and p04 run pair 0 first); E3 / E1 rows refused or
#         returned with no work; controls launch (the instrument sees work).
# Every row records its result; the verdicts are scripts/bwdf3_gpu_post.py's (CPU, after the line).
# Prior art: NumPy's normalize_axis_tuple ("duplicate value in axis") and reductions as the references; the RoPE
# formulation (Su et al., 2021, "RoFormer"); ENG3's call counter (ltrace-style interposition); scripts/bwdf2_step_k2.py
# (this repo), whose structure this follows. Ours: the cases.
import json
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bwdf3_gpu_common as G  # noqa: E402

import bwdf3_p2_cases as C  # noqa: E402

MODES = C.MODES
rows_for, run_row = C.rows_for, C.run_row


def child(mode, arm, rid, out_dir):
    """One row in this process: build the operands (uncounted, synchronized), then count only the call."""
    import ctypes
    import numpy as np
    tc, sha = G.load_arm(arm)
    counted = mode == "count"
    if counted:
        L = ctypes.CDLL(None)
        L.eng3_cudart_count.restype = ctypes.c_ulong
        L.eng3_cudart_name.restype = ctypes.c_char_p
        names = [L.eng3_cudart_name(i).decode() for i in range(L.eng3_cudart_n())]
        counts = lambda: {n: L.eng3_cudart_count(i) for i, n in enumerate(names)}
    row = {"id": rid, "module_sha256": sha}
    with tc.no_grad():
        try:
            build, call = run_row(tc, mode, rid)
            ops = build()
            tc.synchronize()
        except Exception as e:  # noqa: BLE001
            row.update(result="SETUP-ERROR", error=f"{type(e).__name__}: {e}")
            print("ROW " + json.dumps(row))
            return
        c0 = counts() if counted else None
        try:
            r = call(ops)
            tc.synchronize()
            row["result"] = "RETURNED"
            if isinstance(r, dict):
                np.savez(pathlib.Path(out_dir) / f"{rid}.npz", **r)
                row["arrays"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in r.items()}
        except Exception as e:  # noqa: BLE001
            row["result"] = "RAISED"
            row["error"] = f"{type(e).__name__}: {e}"
        if counted:
            c1 = counts()
            row["by_function"] = {n: c1[n] - c0[n] for n in names if c1[n] != c0[n]}
    print("ROW " + json.dumps(row, default=str))


def main():
    if sys.argv[1] == "--child":
        return child(*sys.argv[2:6])
    mode, arm = sys.argv[1], sys.argv[2]
    assert mode in MODES, mode
    out = G.GPU / f"p2_{mode}_{arm}"
    if out.exists():
        raise SystemExit(f"{out} exists; move it aside before a retry")
    out.mkdir(parents=True)
    sha = G.check_arm("tc", arm)
    env = dict(os.environ)
    if mode == "count":
        shim = out / "cudart_count.so"
        cc = shutil.which("cc") or shutil.which("gcc")
        r = subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", str(shim),
                            str(G.ROOT / "tensor_cuda/tests/support/cudart_count.c"), "-ldl"], capture_output=True,
                           text=True, timeout=120)
        if r.returncode != 0:
            raise SystemExit("shim build failed: " + r.stderr[-1000:])
        env["LD_PRELOAD"] = str(shim)
    rows = []
    import time
    t_end = time.monotonic() + 470  # inside the line's inner `timeout 500s`: a slow row cannot take the whole line
    for rid, kind in rows_for(mode):
        left = t_end - time.monotonic()
        if left < 5:
            rows.append({"id": rid, "kind": kind, "result": "NOT-RUN (line budget spent)", "child_exit": None})
            continue
        try:
            p = subprocess.run([sys.executable, "-B", "-P", __file__, "--child", mode, arm, rid, str(out)], env=env,
                               capture_output=True, text=True, timeout=min(120, left))
        except subprocess.TimeoutExpired:  # subprocess.run kills its own child on timeout
            rows.append({"id": rid, "kind": kind, "result": "CHILD-TIMEOUT", "child_exit": 124})
            print(json.dumps(rows[-1]), flush=True)
            continue
        got = [l[4:] for l in p.stdout.splitlines() if l.startswith("ROW ")]
        row = json.loads(got[-1]) if got else {"id": rid, "result": "CHILD-DIED",
                                               "error": f"child exit {p.returncode}: {p.stderr[-800:]}"}
        row["kind"] = kind
        row["child_exit"] = p.returncode
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
    (out / "results.json").write_text(json.dumps({"line": f"p2_{mode}_{arm}", "mode": mode, "arm": arm,
                                                  "module_sha256": sha, "rows": rows}, indent=1, default=str))
    print("DONE", mode, arm, len(rows), "rows")


if __name__ == "__main__":
    main()
