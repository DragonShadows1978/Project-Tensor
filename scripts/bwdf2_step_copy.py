import subprocess
if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")
# BWD-F2 R11 restage (a copy of scripts/bwdf1_step_copy.py: the held-lease check, bwdf2_gpu_common, outputs under
# artifacts/bwdf1/f2/gpu/). BWD-F1 Amendment 1 C3 (staged GPU line, run under compute-sanitizer memcheck by the lead): Voice's two cat_nd cases
# (first operand [1, 512, 0] with a later [1, 1, 1], cat on dim 1; a later operand one frame longer than the first)
# and one mismatch case per other copy op, with CUDA operands that hold data. Build B: record what happens. Build E:
# each refuses (RuntimeError) before any launch, and memcheck reports 0 errors.
# Usage: python3 -B -P scripts/bwdf2_step_copy.py <arm>; writes only artifacts/bwdf1/f2/gpu/copy_<arm>/ (refuses if it
# exists). Prior art: numpy.concatenate / numpy.stack rules; ours: the cases.
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bwdf2_gpu_common as G  # noqa: E402

arm = sys.argv[1]
OUT = G.GPU / f"copy_{arm}"
if OUT.exists():
    raise SystemExit(f"{OUT} exists; move it aside before a retry")
OUT.mkdir(parents=True)
tc, sha = G.load_arm(arm)
import numpy as np  # noqa: E402


def T(shape, dt="float32", salt=0):
    return tc.tensor(G.data(shape, dt, salt), device="cuda", dtype=dt)


CASES = [
    ("cat-voice-zero-first", "cat", lambda: tc.cat([T((1, 512, 0)), T((1, 1, 1), salt=1)], 1)),
    ("cat-voice-longer-later", "cat", lambda: tc.cat([T((1, 3, 5)), T((1, 2, 6), salt=1)], 1)),
    ("cat-dtype", "cat", lambda: tc.cat([T((2, 3)), T((2, 3), "float16", 1)], 0)),
    ("stack-shape", "stack", lambda: tc.stack([T((2, 3)), T((2, 4), salt=1)], 0)),
    ("slice-past-end", "slice", lambda: T((4, 3)).slice(0, 3, 2)),
    ("write_rows-start-negative", "write_rows", lambda: tc.write_rows(T((1, 4, 2)), T((1, 1, 2), salt=1), -1)),
    ("export_rows-rank0", "export_rows", lambda: tc.export_rows(tc.tensor(np.float32(1.0), device="cuda"), 0, 0, 1)),
    ("export_rows-past-end", "export_rows", lambda: tc.export_rows(T((1, 4, 2)), 1, 3, 2)),
    ("splice_rows-extent", "splice_rows", lambda: tc.splice_rows(T((1, 4, 2)), T((1, 1, 3), salt=1), 1, 0, 1)),
    ("evict_rows-drop-past-tail", "evict_rows", lambda: tc.evict_rows(T((1, 4, 2)), 1, 3, 2)),
    ("export_rope_rows-sin-short", "export_rope_rows",
     lambda: tc.export_rope_rows(T((1, 4, 2)), T((8, 2), salt=1), T((1, 2), salt=2), 1, 0, 2, 0)),
    ("swap_row_pairs-pair1-extent", "swap_row_pairs_with_rope",
     lambda: tc.swap_row_pairs_with_rope([T((1, 4, 2)), T((1, 4, 2), salt=1)], [T((1, 4, 2), salt=2),
                                         T((1, 4, 2), salt=3)], [T((1, 1, 2), salt=4), T((1, 1, 3), salt=5)],
                                         [T((1, 1, 2), salt=6), T((1, 1, 2), salt=7)], T((8, 2), salt=8),
                                         T((8, 2), salt=9), 1, 1, 0, 1, 0)),
]
GRAD_FREE = {"write_rows", "export_rows", "splice_rows", "evict_rows", "export_rope_rows", "swap_row_pairs_with_rope"}
rows = []
for cid, op, fn in CASES:
    row = {"id": cid, "op": op}
    try:
        if op in GRAD_FREE:
            with tc.no_grad():
                r = fn()
        else:
            r = fn()
        tc.synchronize()
        row["result"] = "RETURNED"
        row["shape"] = None if r is None or isinstance(r, (tuple, list)) else list(r.shape)
    except Exception as e:  # noqa: BLE001
        row["result"] = "RAISED"
        row["error"] = f"{type(e).__name__}: {e}"
    rows.append(row)
    print(json.dumps(row), flush=True)
refused = all(r["result"] == "RAISED" and r["error"].startswith("RuntimeError: ") and "CUDA error" not in r["error"]
              and "illegal" not in r["error"] for r in rows)
(OUT / "results.json").write_text(json.dumps({"line": f"copy_{arm}", "arm": arm, "module_sha256": sha, "rows": rows,
                                              "all_refused_on_host": refused}, indent=1))
print("VERDICT", "ALL-REFUSED" if refused else "NOT-ALL-REFUSED (see rows)")
