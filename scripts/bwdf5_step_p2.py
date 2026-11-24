import subprocess
if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")
# BWD-F5 P2 (staged GPU line; repair of scripts/bwdf3_step_p2.py): the E1-E3 cases that only the GPU can show, one mode
# per line, each row in its own child process AND each child under its own compute-sanitizer memcheck. This process
# (the step) runs inside the lease and OUTSIDE the sanitizer; the line's command has no sanitizer. Usage:
#   python3 -B -P scripts/bwdf5_step_p2.py <mode> <arm>      mode: e1 | e2 | e3 | canary; arm: B | F2 | F4
# writes only artifacts/bwdf1/f5/gpu/p2_<mode>_<arm>/ (refuses to start if it exists): per row <id>.row.json,
# <id>.memcheck.log, <id>.npz (returned arrays), and results.json. The child (`--child`) starts with the same lease
# check (line 2 of this file), then runs one row: operands are CUDA tensors with data unless named empty.
#   e1    RoPE read spans on reading launches: export_rope_rows / export_row_pairs / swap / arena with a short sin, a
#         negative pos0 or a short cos are refused before any work; the empty launches with CUDA tables return their
#         shapes; controls (covering tables) keep their values (B vs F4 bitwise).
#   e2    table layout: [T, D, 1] / [T, D, 1, 1] tables give results bitwise equal to the [T, D] tables in the same
#         build (rope_apply, export_rope_rows, swap); a [T, D, 2] table is refused.
#   e3    reduction axes: sum([0, 0]) on [2] (the review's case: the tail kernel reads 4 elements of a 2-element buffer
#         on Build F2), sum([8]) on rank 1, mean([0, 0]), softmax(-3), argmax(2) refused before any work; valid-axis
#         controls keep their values (B vs F4 bitwise).
#   canary  ONE control row of e1 (the rope_apply control) on the arm, under the per-row sanitizer: the lead sees at
#         once whether a child survives memcheck (exit 0, a memcheck log with ERROR SUMMARY: 0 errors, a RETURNED row).
# `count` is not restaged (BWD-F3's count lines ran without a sanitizer and exited 0).
# Every row's verdict is scripts/bwdf5_gpu_post.py's (CPU, after the line); the memcheck number is read per row.
# Prior art: compute-sanitizer's --log-file option (NVIDIA documentation); one process per test row (pytest-forked
# style); NumPy's normalize_axis_tuple and reductions as the references; the RoPE formulation (Su et al., 2021,
# "RoFormer"); scripts/bwdf3_step_p2.py (this repo), which this repairs. Ours: the per-row sanitizer and the cases.
import json
import os
import pathlib
import signal
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bwdf5_gpu_common as G  # noqa: E402
import bwdf5_p2_cases as C  # noqa: E402
import bwdf5_p2_run as R  # noqa: E402

run_row = C.run_row
SELF = pathlib.Path(__file__).resolve()


def child(mode, arm, rid, out_dir):
    """One row in this process: build the operands (synchronized), then call. Writes <rid>.row.json and prints ROW."""
    import numpy as np
    tc, sha = G.load_arm(arm)
    row = {"id": rid, "module_sha256": sha}
    out = pathlib.Path(out_dir)

    def finish():
        tmp = out / f"{rid}.row.json.tmp"
        tmp.write_text(json.dumps(row, default=str))
        os.replace(tmp, out / f"{rid}.row.json")
        print("ROW " + json.dumps(row, default=str))

    with tc.no_grad():
        try:
            build, call = run_row(tc, R.case_mode(mode), rid)
            ops = build()
            tc.synchronize()
        except Exception as e:  # noqa: BLE001
            row.update(result="SETUP-ERROR", error=f"{type(e).__name__}: {e}")
            return finish()
        try:
            r = call(ops)
            tc.synchronize()
            row["result"] = "RETURNED"
            if isinstance(r, dict):
                np.savez(out / f"{rid}.npz", **r)
                row["arrays"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in r.items()}
        except Exception as e:  # noqa: BLE001
            row["result"] = "RAISED"
            row["error"] = f"{type(e).__name__}: {e}"
    finish()


def child_argv(mode, arm, out):
    # `python3 -B -P` as the lead's lines run (L109); the sanitizer prefix is added by the runner
    return lambda rid: [sys.executable, "-B", "-P", str(SELF), "--child", mode, arm, rid, str(out)]


def main():
    if sys.argv[1] == "--child":
        return child(*sys.argv[2:6])
    mode, arm = sys.argv[1], sys.argv[2]
    assert mode in R.LINE_MODES, mode
    assert arm in G.ARMS, arm
    out = G.GPU / f"p2_{mode}_{arm}"
    if out.exists():
        raise SystemExit(f"{out} exists; move it aside before a retry")
    out.mkdir(parents=True)
    sha = G.check_arm("tc", arm)
    # `timeout` (the line's inner limit) sends SIGTERM to this process only: turn it into an exit so run_one's cleanup
    # takes the running child's process group down with us instead of orphaning a sanitized child
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    rows = R.run_rows(mode, arm, out, R.SANITIZER, child_argv(mode, arm, out),
                      emit=lambda r: print(json.dumps(r, default=str), flush=True))
    (out / "results.json").write_text(json.dumps({"line": f"p2_{mode}_{arm}", "mode": mode, "arm": arm,
                                                  "module_sha256": sha, "sanitizer": R.SANITIZER + ["--log-file",
                                                                                                    "<row log>"],
                                                  "rows": rows}, indent=1, default=str))
    print("DONE", mode, arm, len(rows), "rows")


if __name__ == "__main__":
    main()
