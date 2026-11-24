import subprocess
if subprocess.run(["forge-gpu-lease-check"]).returncode != 0: raise SystemExit("not inside a forge-gpu-run lease")
# BWD-F1 G3 (staged GPU line): zero-extent gradients through reduce_to. Every differentiable op of the census on a
# zero-extent broadcast ([4, 0] against [4, 1]), each dtype, each case in its own child process (so a device fault on
# Build B or a mutant is recorded per case, not carried into the next one); gradients compared with NumPy zeros of
# each operand's shape (scripts/bwdf1_gpu_common.py case_verdict). Arms: B (record), E (bar), m03_reduce_gt1 (G5).
# Usage: python3 -B -P scripts/bwdf1_step_g3.py <arm>; writes only artifacts/bwdf1/gpu/g3_<arm>/ (refuses if it
# exists). Prior art: PyTorch's sum_to_size semantics; NumPy's empty sum. Ours: the per-case isolation.
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bwdf1_gpu_common as G  # noqa: E402

CHILD = r'''
import subprocess
if subprocess.run(["forge-gpu-lease-check"]).returncode != 0: raise SystemExit("not inside a forge-gpu-run lease")
import json, sys
sys.path.insert(0, sys.argv[1])
import bwdf1_gpu_common as G
tc, _ = G.load_arm(sys.argv[2])
row, _ = G.run_case(tc, json.loads(sys.argv[3]))
print("ROW " + json.dumps(row))
'''

arm = sys.argv[1]
OUT = G.GPU / f"g3_{arm}"
if OUT.exists():
    raise SystemExit(f"{OUT} exists; move it aside before a retry")
OUT.mkdir(parents=True)
G.check_arm("tc", arm)
rows = []
for c in G.cases():
    if not c["grad"]:
        continue
    p = subprocess.run([sys.executable, "-B", "-P", "-c", CHILD, str(HERE), arm, json.dumps(c)],
                       capture_output=True, text=True, timeout=120)
    got = [l[4:] for l in p.stdout.splitlines() if l.startswith("ROW ")]
    row = json.loads(got[-1]) if got else {"id": c["id"], "result": "ERROR",
                                           "error": f"child exit {p.returncode}: {p.stderr[-600:]}"}
    row["child_exit"] = p.returncode
    row["verdict"] = G.case_verdict(row)
    rows.append(row)
    print(json.dumps(row), flush=True)
verdict = "PASS" if rows and all(r["verdict"] == "PASS" for r in rows) else "FAIL"
(OUT / "results.json").write_text(json.dumps({"line": f"g3_{arm}", "arm": arm, "rows": rows, "verdict": verdict},
                                             indent=1))
print("VERDICT", verdict, sum(r["verdict"] == "PASS" for r in rows), "/", len(rows))
