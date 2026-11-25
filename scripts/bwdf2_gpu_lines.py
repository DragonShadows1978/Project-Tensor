"""BWD-F2 staged GPU lines: R11's restage of BWD-F1's 35 lines against Build F2 (Build B kept as the baseline), K2's
GPU-only rows, and the GPU mutants. The table, the pre-lease gate, GPU_LINES_F2.json and the dry runs.

  python3 -B -P scripts/bwdf2_gpu_lines.py gate <line>     # pre-lease (L103/L119): exit 0 ok, 3 refused
  python3 -B -P scripts/bwdf2_gpu_lines.py write            # artifacts/bwdf1/f2/GPU_LINES_F2.json
  python3 -B -P scripts/bwdf2_gpu_lines.py dryrun           # each line once, exactly as written, FORGE_GPU_DRY_RUN=1

Each line is one shell command, run from the worktree root:
  python3 -B -P scripts/bwdf2_gpu_lines.py gate <line> && forge-gpu-run --min <m> --max 540 --label bwdf2-<line> --
      timeout <t>s [compute-sanitizer --tool memcheck --log-file <log>] python3 -B -P <step> <args>
The gate reads files only (no engine import, no CUDA call): every loaded build's module sha256 against its receipt
(frozen package and build copy) and the line's write-once paths, which must not exist yet. Every step script starts
with `forge-gpu-lease-check --require-held` (L125; the lines are listed in GPU_LINES_F2.json lease_check_lines).
Prior art: BWD-F1's and ENG4's staged-line patterns (this repo); no other prior art known to me.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bwdf2_gpu_common as G  # noqa: E402

GPU = "artifacts/bwdf1/f2/gpu"
G4_CHUNKS = 12
K2_ARMS = {"f01": ("B", "E", "F2", "n03_sin_span_unchecked"), "f04": ("B", "E", "F2", "n04_ring_no_modulo"),
           "f07": ("B", "E", "F2", "n05_condition_as_data_dtype"), "vals": ("B", "E", "F2"),
           "count": ("B", "E", "F2")}


def lines():
    L = {}
    for arm in ("B", "F2"):   # R11: BWD-F1's lines, Build E replaced by Build F2
        for tag, script in (("f01", f"artifacts/bwdf1/f2/gpu_lines/{arm}/f01_halfprec_zero_inner.py"),
                            ("f03", f"artifacts/bwdf1/f2/gpu_lines/{arm}/f03_zero_batch.py")):
            L[f"g2_{tag}_{arm}"] = dict(gate="R11 (BWD-F1 G2)", arm=arm, mods=[("tc", arm)], memcheck=True, min=240,
                                        timeout=500, argv=[script], out=f"{GPU}/g2_{tag}_{arm}",
                                        peak_host_bytes=4 << 30)
        L[f"g2_direct_{arm}"] = dict(gate="R11 (BWD-F1 G2)", arm=arm, mods=[("tc", arm)], memcheck=True, min=300,
                                     timeout=500, argv=["scripts/bwdf2_step_g2_direct.py", arm],
                                     out=f"{GPU}/g2_direct_{arm}", peak_host_bytes=4 << 30)
        L[f"copy_{arm}"] = dict(gate="R11 (BWD-F1 C3)", arm=arm, mods=[("tc", arm)], memcheck=True, min=240,
                                timeout=500, argv=["scripts/bwdf2_step_copy.py", arm], out=f"{GPU}/copy_{arm}",
                                peak_host_bytes=4 << 30)
    for arm in ("B", "F2", "m03_reduce_gt1"):
        L[f"g3_{arm}"] = dict(gate="R11 (BWD-F1 G3)" if arm in ("B", "F2") else "R11 (BWD-F1 G5 m03)", arm=arm,
                              mods=[("tc", arm)], memcheck=False, min=300, timeout=500,
                              argv=["scripts/bwdf2_step_g3.py", arm], out=f"{GPU}/g3_{arm}", peak_host_bytes=3 << 30)
    for arm in ("B", "F2"):
        for i in range(1, G4_CHUNKS + 1):
            L[f"g4_{arm}_{i:02d}"] = dict(gate="K3 GPU half (R11: BWD-F1 G4)", arm=arm, mods=[("tc", arm), ("ts", arm)],
                                          memcheck=False, min=480, timeout=500,
                                          argv=["scripts/bwdf2_step_g4.py", f"gpu_{arm}", arm, "330"],
                                          out=f"artifacts/bwdf1/f2/g4/gpu_{arm}/part_<n> (created by the runner, "
                                              f"write-once per part)", peak_host_bytes=6 << 30, series=True)
    for mode, arms in K2_ARMS.items():
        for arm in arms:
            gate = ("K4 GPU mutant" if arm.startswith("n0") else "K2")
            L[f"k2_{mode}_{arm}"] = dict(gate=gate, arm=arm, mods=[("tc", arm)], memcheck=mode != "count", min=240,
                                         timeout=500, argv=["scripts/bwdf2_step_k2.py", mode, arm],
                                         out=f"{GPU}/k2_{mode}_{arm}", peak_host_bytes=3 << 30)
    for k, v in L.items():
        mc = (["compute-sanitizer", "--tool", "memcheck", "--log-file", f"{GPU}/{k}.memcheck.log"]
              if v["memcheck"] else [])
        v["command"] = (f"python3 -B -P scripts/bwdf2_gpu_lines.py gate {k} && forge-gpu-run --min {v['min']} "
                        f"--max 540 --label bwdf2-{k} -- timeout {v['timeout']}s "
                        + " ".join(mc + ["python3", "-B", "-P"] + v["argv"]))
        v["write_once"] = ([] if v.get("series") else [v["out"]]) + ([f"{GPU}/{k}.memcheck.log"] if v["memcheck"]
                                                                     else [])
    return L


def gate(line):
    v = lines()[line]
    for pkg, arm in v["mods"]:
        try:
            got = G.check_arm(pkg, arm)
        except (SystemExit, OSError) as e:
            print(f"bwdf2 gate {line}: {e}", file=sys.stderr)
            return 3
        print(f"bwdf2 gate {line}: {pkg} {arm} sha256 {got} matches its receipt")
    for p in v["write_once"]:
        if (ROOT / p).exists():
            print(f"bwdf2 gate {line}: {p} exists; move it aside before a retry", file=sys.stderr)
            return 3
    prog = ROOT / f"artifacts/bwdf1/f2/g4/gpu_{v['arm']}/progress.json"
    if v.get("series") and prog.exists() and json.loads(prog.read_text()).get("complete"):
        print(f"bwdf2 gate {line}: series gpu_{v['arm']} complete; the step exits 0 at once")
    return 0


LEASE_FILES = ["scripts/bwdf2_step_g2_direct.py", "scripts/bwdf2_step_g3.py", "scripts/bwdf2_step_copy.py",
               "scripts/bwdf2_step_g4.py", "scripts/bwdf2_step_k2.py",
               "artifacts/bwdf1/f2/gpu_lines/B/f01_halfprec_zero_inner.py",
               "artifacts/bwdf1/f2/gpu_lines/B/f03_zero_batch.py",
               "artifacts/bwdf1/f2/gpu_lines/F2/f01_halfprec_zero_inner.py",
               "artifacts/bwdf1/f2/gpu_lines/F2/f03_zero_batch.py"]


def lease_lines():
    """L125: file:line of each step script's held-lease check (and of the child scripts' checks)."""
    out = {}
    for f in LEASE_FILES:
        src = (ROOT / f).read_text().splitlines()
        hits = [i + 1 for i, l in enumerate(src) if 'forge-gpu-lease-check", "--require-held"' in l]
        assert src[0] == "import subprocess" and hits and hits[0] == 2, (f, hits)
        out[f] = [f"{f}:{h}" for h in hits]
    return out


def write():
    L = lines()
    doc = {"order": "BWD-F2 (R11 restage, K2, K3 GPU half, K4 GPU mutants)", "cwd": "worktree root", "lines": L,
           "n_lines": len(L),
           "lead_records_per_line": "forge-gpu-run exit, the inner timeout exit (124 = timeout, never a pass), the run "
                                    "log's sha256 (L45)",
           "exit_semantics": "gate 3 = refused before the lease (module sha or write-once path); forge-gpu-run 75/76 "
                             "= no window or lease (nothing ran); 124 = timeout; step 0 = completed, verdict from "
                             "scripts/bwdf2_gpu_post.py; a G4 step exits 0 at once when its series is complete",
           "rlimit_as": "none set (L114); the lead runs each line in a scope with MemoryMax=12G, OOMPolicy=continue",
           "post_device_followups_cpu": [
               "python3 -B -P scripts/bwdf2_gpu_post.py line <line>   (each non-G4 line -> "
               "artifacts/bwdf1/f2/gpu/<line>.post.json)",
               "python3 -B -P scripts/bwdf2_gpu_post.py controls       (accepted results, Build B vs Build F2 bitwise)",
               "python3 -B -P scripts/bwdf2_gpu_post.py g4             (K3 GPU per-ID, B vs F2)"],
           "lease_check_lines": lease_lines(),
           "write_once_paths": {k: v["write_once"] for k, v in L.items()},
           "copied_audit_scripts": "artifacts/bwdf1/f2/gpu_lines/<arm>/f01_halfprec_zero_inner.py and f03_zero_batch.py "
                                   "are BWD-F1's copies with three lines changed (the held-lease check; ROOT one level "
                                   "deeper; OUT under artifacts/bwdf1/f2/gpu/); aud1_verdicts.py is byte-identical; "
                                   "aud1_gpu_load.py keeps the audit's API with the module pin set to Build B / F2"}
    (ROOT / "artifacts/bwdf1/f2/GPU_LINES_F2.json").write_text(json.dumps(doc, indent=1))
    print("lines", len(L))


def dryrun():
    out = []
    for k, v in lines().items():
        env = dict(os.environ, FORGE_GPU_DRY_RUN="1")
        p = subprocess.run(["bash", "-c", v["command"]], cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=120)
        out.append({"line": k, "command": v["command"], "exit": p.returncode, "stdout": p.stdout[-600:],
                    "stderr": p.stderr[-600:]})
        print(k, p.returncode, flush=True)
    d = ROOT / "artifacts/bwdf1/f2/gpu_dry"
    d.mkdir(exist_ok=True)
    (d / "DRY_RUN.json").write_text(json.dumps(out, indent=1))
    created = [p for v in lines().values() for p in v["write_once"] if (ROOT / p).exists()]
    print("dry-run exits:", sorted({r["exit"] for r in out}), "write-once paths created:", created)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gate":
        sys.exit(gate(sys.argv[2]))
    if mode == "write":
        write()
    elif mode == "dryrun":
        dryrun()
