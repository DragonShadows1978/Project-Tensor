"""BWD-F5 staged GPU lines: the repaired P2 rows (E1-E3) on Builds B, F2 and F4, plus one canary line. The table, the
pre-lease gate, GPU_LINES_F5.json and the dry runs. Outputs under artifacts/bwdf1/f5/.

  python3 -B -P scripts/bwdf5_gpu_lines.py gate <line>     # pre-lease (L103/L119): exit 0 ok, 3 refused
  python3 -B -P scripts/bwdf5_gpu_lines.py write           # artifacts/bwdf1/f5/GPU_LINES_F5.json (no dry-run record)
  python3 -B -P scripts/bwdf5_gpu_lines.py dryrun          # each line once, exactly as written, FORGE_GPU_DRY_RUN=1,
                                                           # then rewrites GPU_LINES_F5.json with the exits

Each line is one shell command, run from the worktree root:
  python3 -B -P scripts/bwdf5_gpu_lines.py gate <line> && forge-gpu-run --min 240 --max 540 --label bwdf5-<line> --
      timeout 500s python3 -B -P scripts/bwdf5_step_p2.py <mode> <arm>
There is NO sanitizer in the line (BWD-F3's `--target-processes all` wrapper is what killed every child): the step runs
inside the lease and starts one `compute-sanitizer --tool memcheck --log-file <row log>` per row child.
The gate reads files only (no engine import, no CUDA call): the arm's module sha256 against its receipt (frozen package
and build copy) and the line's write-once paths, which must not exist yet. The step and each of its children start with
`forge-gpu-lease-check --require-held` (scripts/bwdf5_step_p2.py:2; the child re-executes that file; L125).
Prior art: BWD-F3's staged-line pattern (scripts/bwdf3_gpu_lines.py, this repo), copied. No other prior art known to me.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bwdf5_gpu_common as G  # noqa: E402
import bwdf5_p2_run as R  # noqa: E402

GPU = "artifacts/bwdf1/f5/gpu"
ORDER = ("canary_F4", "e1_B", "e1_F2", "e1_F4", "e2_B", "e2_F2", "e2_F4", "e3_B", "e3_F2", "e3_F4")  # canary first


def lines():
    L = {}
    for key in ORDER:
        mode, arm = key.split("_")
        k = f"p2_{key}"
        out = f"{GPU}/{k}"
        L[k] = dict(gate="P2 repaired harness (canary)" if mode == "canary" else "P2 (BWD-F3, restaged)", arm=arm,
                    mode=mode, mods=[("tc", arm)], min=240, timeout=500, argv=["scripts/bwdf5_step_p2.py", mode, arm],
                    out=out, rows=[rid for rid, _ in R.rows_for(mode)], peak_host_bytes=3 << 30)
        L[k]["command"] = (f"python3 -B -P scripts/bwdf5_gpu_lines.py gate {k} && forge-gpu-run --min 240 --max 540 "
                           f"--label bwdf5-{k} -- timeout 500s python3 -B -P scripts/bwdf5_step_p2.py {mode} {arm}")
        # the directory is write-once; the per-row files below are created inside it
        L[k]["write_once"] = [out]
        L[k]["write_once_inside"] = [f"{out}/results.json"] + [f"{out}/{rid}.{ext}" for rid in L[k]["rows"]
                                                              for ext in ("memcheck.log", "row.json")]
    return L


def gate(line):
    v = lines()[line]
    for pkg, arm in v["mods"]:
        try:
            got = G.check_arm(pkg, arm)
        except (SystemExit, OSError) as e:
            print(f"bwdf5 gate {line}: {e}", file=sys.stderr)
            return 3
        print(f"bwdf5 gate {line}: {pkg} {arm} sha256 {got} matches its receipt")
    for p in v["write_once"]:
        if (ROOT / p).exists():
            print(f"bwdf5 gate {line}: {p} exists; move it aside before a retry", file=sys.stderr)
            return 3
    return 0


def lease_lines():
    """L125: file:line of the step's (and, by re-execution, each child's) held-lease check."""
    f = "scripts/bwdf5_step_p2.py"
    src = (ROOT / f).read_text().splitlines()
    hits = [i + 1 for i, l in enumerate(src) if 'forge-gpu-lease-check", "--require-held"' in l]
    assert src[0] == "import subprocess" and hits == [2], (f, hits)
    return {f: [f"{f}:2"], "child": f"{f}:2 (the child is this same file started with --child; it checks first)"}


def write(dry=None):
    L = lines()
    if dry:
        for k, d in dry.items():
            L[k]["dry_run"] = d
    doc = {"order": "BWD-F5 (repair of BWD-F3's P2 harness: per-row sanitizer; P2 rows restaged on Builds B, F2, F4; "
                    "canary first)",
           "cwd": "worktree root", "lines": L, "n_lines": len(L), "run_order": list(L),
           "lead_records_per_line": "forge-gpu-run exit, the inner timeout exit (124 = timeout, never a pass), the run "
                                    "log's sha256, and each row's memcheck log sha256 (results.json rows: "
                                    "memcheck_log_sha256) (L45)",
           "exit_semantics": "gate 3 = refused before the lease (module sha or write-once path); forge-gpu-run 75/76 "
                             "= no window or lease (nothing ran); 78 = FORGE_GPU_DRY_RUN; 124 = timeout; step 0 = "
                             "completed, verdicts from scripts/bwdf5_gpu_post.py",
           "rlimit_as": "none set (L114); the lead runs each line in a scope with MemoryMax=12G, OOMPolicy=continue",
           "post_device_followups_cpu": [
               "python3 -B -P scripts/bwdf5_gpu_post.py line <line>   (each line -> artifacts/bwdf1/f5/gpu/<line>.post.json)",
               "python3 -B -P scripts/bwdf5_gpu_post.py controls      (accepted results, Build B vs Build F4 bitwise)"],
           "lease_check_lines": lease_lines(),
           "write_once_paths": {k: v["write_once"] for k, v in L.items()},
           "modules": {a: {"receipt": G.ARMS[a][0], "build_dir": str(G.ARMS[a][1].relative_to(G.ROOT.parent)),
                           "module_sha256": json.loads(G.receipt("tc", a).read_text())["module_sha256"]}
                       for a in G.ARMS},
           "note": "BWD-F3's count lines (no sanitizer) exited 0 and are not restaged; the canary runs first."}
    (ROOT / "artifacts/bwdf1/f5/GPU_LINES_F5.json").write_text(json.dumps(doc, indent=1))
    print("lines", len(L))


def dryrun():
    out, dry = [], {}
    for k, v in lines().items():
        env = dict(os.environ, FORGE_GPU_DRY_RUN="1")
        p = subprocess.run(["bash", "-c", v["command"]], cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=120)
        out.append({"line": k, "command": v["command"], "exit": p.returncode, "stdout": p.stdout[-600:],
                    "stderr": p.stderr[-600:]})
        dry[k] = {"exit": p.returncode}
        print(k, p.returncode, flush=True)
    d = ROOT / "artifacts/bwdf1/f5/gpu_dry"
    d.mkdir(exist_ok=True)
    (d / "DRY_RUN.json").write_text(json.dumps(out, indent=1))
    created = [p for v in lines().values() for p in v["write_once"] if (ROOT / p).exists()]
    print("dry-run exits:", sorted({r["exit"] for r in out}), "write-once paths created:", created)
    write(dry)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gate":
        sys.exit(gate(sys.argv[2]))
    if mode == "write":
        write()
    elif mode == "dryrun":
        dryrun()
